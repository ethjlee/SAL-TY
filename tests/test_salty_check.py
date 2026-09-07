import contextlib
import io
import json
import os
import subprocess
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

import salty_check as checker


class CheckerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.folder = self.root / "images" / "000001"
        self.folder.mkdir(parents=True)
        self.metadata = self.root / "metadata"
        self.metadata.mkdir()
        self.meta_path = self.metadata / "000001.json"
        self.base_meta = dict(
            index=1, panoid="test-pano", pano_lat=37.0, pano_lon=-122.0,
            original_lat=37.0, original_lon=-122.0,
            headings=[0, 90, 180, 270], view_resolution="1024x1024",
            view_fov=90, copyright="Google", country_code="US",
        )
        self.write_meta()
        checker._init_worker(None, {1: "test-pano"})
        self.addCleanup(checker._init_worker, None, None)

    def write_meta(self, **changes):
        self.meta_path.write_text(json.dumps(dict(self.base_meta, **changes)))

    def scan(self):
        return checker._process_folder(self.folder, True, self.metadata)

    def save_image(self, pixels, name="000.jpg"):
        Image.fromarray(pixels).save(self.folder / name, quality=90)

    def write_archive(self, members):
        """Write ordered tar members; a None payload represents a directory."""
        archive_path = self.root / "fixture.tar"
        with tarfile.open(archive_path, "w") as archive:
            for name, raw in members:
                info = tarfile.TarInfo(name)
                if raw is None:
                    info.type = tarfile.DIRTYPE
                    archive.addfile(info)
                else:
                    info.size = len(raw)
                    archive.addfile(info, io.BytesIO(raw))
        return archive_path

    def test_valid_metadata(self):
        result = self.scan()
        self.assertIsNone(result["corrupt_meta"])
        self.assertFalse(result["meta_value_issues"])
        self.assertFalse(result["bad_coords"])
        self.assertEqual(result["panoid"], "test-pano")

    def test_null_required_fields_are_reported(self):
        for field in checker.REQUIRED_META_FIELDS:
            with self.subTest(field=field):
                self.write_meta(**{field: None})
                result = self.scan()
                self.assertIsNone(result["corrupt_meta"])
                self.assertTrue(result["meta_value_issues"] or result["meta_index_mismatch"])

    def test_invalid_panoid_types(self):
        for value in [[], ["bad"], {"id": "bad"}, 42, True, "", "   "]:
            with self.subTest(value=value):
                self.write_meta(panoid=value)
                result = self.scan()
                self.assertIsNone(result["panoid"])
                self.assertTrue(result["meta_value_issues"])
                self.assertIsNone(result["corrupt_meta"])

    def test_malformed_panoid_survives_process_pool_and_exports(self):
        self.write_meta(panoid=["bad"])
        findings = checker._scan_all_folders(
            [self.folder], True, self.metadata, None, {1: "test-pano"}, 1,
        )
        self.assertTrue(findings.meta_value_issues)
        self.assertFalse(findings.panoid_map)
        sections = checker._build_export_sections(findings, checker.DerivedFindings())
        path = self.root / "flagged.txt"
        checker.write_flagged_export(path, sections)
        self.assertIn("meta_value_issues: panoid", path.read_text(encoding="utf-8"))

    def test_fractional_and_nonfinite_json_indices(self):
        for value in [1.9, "1.9", True, float("inf"), float("nan")]:
            with self.subTest(value=value):
                self.write_meta(index=value)
                self.assertIsNotNone(self.scan()["meta_index_mismatch"])
        for value in [1, "1", 1.0, "1.0"]:
            with self.subTest(value=value):
                self.write_meta(index=value)
                self.assertIsNone(self.scan()["meta_index_mismatch"])

    def test_metadata_checks_accumulate_independent_findings(self):
        self.write_meta(
            index=2, panoid="other-pano", headings=[0], view_resolution="32x32",
            view_fov="invalid", copyright="Other", country_code="CA", date="2026/01",
            pano_lat=45.0, original_lat=37.01,
        )
        checker._init_worker({1: (37.0, -122.0)}, {1: "test-pano"})
        result = self.scan()
        self.assertIsNone(result["corrupt_meta"])
        self.assertEqual(result["meta_index_mismatch"], (1, 2))
        self.assertEqual(result["meta_value_issues"], [
            (1, "headings", [0, 90, 180, 270], [0]),
            (1, "view_resolution", "1024x1024", "32x32"),
            (1, "view_fov", 90.0, "invalid"),
        ])
        self.assertEqual(result["copyright_issues"], [(1, "Other")])
        self.assertEqual(result["country_code_issues"], [(1, "CA")])
        self.assertEqual(result["bad_dates"], [(1, "2026/01")])
        self.assertEqual(result["panoid_csv_mismatches"], [(1, "test-pano", "other-pano")])
        self.assertEqual(result["coord_mismatch"], (1, 37.01, -122.0, 37.0, -122.0))
        self.assertGreater(result["pano_distance_issue"][1], checker.MAX_PANO_DISTANCE_M)
        self.assertEqual(result["outside_california"], [(1, "pano", 45.0, -122.0)])

    def test_non_object_metadata_is_reported_as_corrupt(self):
        for raw in [b"{", b"null", b"[]", b"42"]:
            with self.subTest(raw=raw):
                self.meta_path.write_bytes(raw)
                result = self.scan()
                self.assertEqual(result["corrupt_meta"][0], 1)
                self.assertEqual(result["disk_bytes"], len(raw))
                self.assertFalse(result["missing_meta"])

    def test_invalid_csv_indices_do_not_drop_valid_rows_or_truncate(self):
        path = self.root / "completed.csv"
        path.write_text("index,panoid\n1,one\n1.9,wrong\n2,two\ninf,bad\n,missing\ntext,bad\n")
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            unique, raw = checker.load_csv_indices([path])
            panoids = checker.load_completed_panoids([path])
        self.assertEqual(unique, {1, 2})
        self.assertEqual(raw, [1, 2])
        self.assertEqual(panoids, {1: "one", 2: "two"})
        self.assertIn("4 non-integer", output.getvalue())

    def test_source_csv_indices_are_not_truncated(self):
        path = self.root / "source.csv"
        path.write_text("index,lat,lon\n1,37,-122\n1.9,38,-123\n2,36,-121\n")
        with contextlib.redirect_stdout(io.StringIO()):
            coords = checker.load_source_coords(path)
        self.assertEqual(coords, {1: (37., -122.), 2: (36., -121.)})

    def test_extra_jpeg_is_reported_and_exported(self):
        rng = np.random.default_rng(42)
        for name in sorted(checker.EXPECTED_IMAGES | {"extra.jpg"}):
            self.save_image(rng.integers(0, 256, (32, 32, 3), dtype=np.uint8), name)
        result = self.scan()
        self.assertIsNone(result["incomplete"])
        self.assertEqual(result["unexpected_files"], [(1, "extra.jpg")])
        findings = checker.ScanFindings(unexpected_files=result["unexpected_files"])
        sections = dict(checker._build_export_sections(findings, checker.DerivedFindings()))
        self.assertEqual(sections["unexpected_files"], [(1, "extra.jpg")])

    def test_extra_jpeg_does_not_replace_missing_view(self):
        self.save_image(np.zeros((32, 32, 3), dtype=np.uint8), "extra.jpg")
        self.assertEqual(self.scan()["incomplete"], (1, sorted(checker.EXPECTED_IMAGES)))

    def test_uniform_colors_are_blank(self):
        for color in [(255, 0, 0), (0, 255, 0), (0, 0, 255), (128, 128, 128)]:
            with self.subTest(color=color):
                pixels = np.empty((1024, 1024, 3), dtype=np.uint8)
                pixels[:] = color
                self.save_image(pixels)
                result = self.scan()
                self.assertTrue(result["blank_imgs"])
                self.assertFalse(result["blurry_imgs"])
                self.assertFalse(result["corrupt_imgs"])

    def test_detailed_view_is_not_blank_or_blurry(self):
        rng = np.random.default_rng(42)
        self.save_image(rng.integers(0, 256, (1024, 1024, 3), dtype=np.uint8))
        result = self.scan()
        self.assertFalse(result["blank_imgs"])
        self.assertFalse(result["blurry_imgs"])

    def test_detail_in_one_channel_is_enough_to_avoid_blank_flag(self):
        pixels = np.zeros((1024, 1024, 3), dtype=np.uint8)
        pixels[:, :, 0] = np.arange(1024, dtype=np.uint16)[None, :] % 256
        self.save_image(pixels)
        self.assertFalse(self.scan()["blank_imgs"])

    def test_grayscale_blank_image_does_not_crash(self):
        self.save_image(np.full((32, 32), 128, dtype=np.uint8))
        result = self.scan()
        self.assertTrue(result["blank_imgs"])
        self.assertFalse(result["corrupt_imgs"])

    def test_jpeg_trailing_bytes_are_not_truncation(self):
        rng = np.random.default_rng(42)
        self.save_image(rng.integers(0, 256, (1024, 1024, 3), dtype=np.uint8))
        path = self.folder / "000.jpg"
        original = path.read_bytes()
        for suffix in [b"", b"trailing data", b"\x00" * 128]:
            with self.subTest(suffix=suffix):
                path.write_bytes(original + suffix)
                result = self.scan()
                self.assertFalse(result["truncated_imgs"])
                self.assertFalse(result["corrupt_imgs"])
                self.assertEqual(result["images_ok"], 1)

    def test_truncated_jpeg_is_still_rejected(self):
        rng = np.random.default_rng(42)
        self.save_image(rng.integers(0, 256, (1024, 1024, 3), dtype=np.uint8))
        path = self.folder / "000.jpg"
        original = path.read_bytes()
        for raw in [original[:-2], original[:len(original) // 2]]:
            with self.subTest(length=len(raw)):
                path.write_bytes(raw)
                result = self.scan()
                self.assertTrue(result["truncated_imgs"])
                self.assertTrue(result["corrupt_imgs"])
                self.assertEqual(result["images_ok"], 0)

    def test_end_marker_in_comment_cannot_hide_truncation(self):
        rng = np.random.default_rng(42)
        self.save_image(rng.integers(0, 256, (1024, 1024, 3), dtype=np.uint8))
        path = self.folder / "000.jpg"
        original = path.read_bytes()
        # JPEG COM segment containing marker-like bytes, followed by a scan
        # whose real end marker is missing.
        path.write_bytes(original[:2] + b"\xff\xfe\x00\x04\xff\xd9" + original[2:-2])
        result = self.scan()
        self.assertTrue(result["truncated_imgs"])
        self.assertTrue(result["corrupt_imgs"])

    def test_duplicate_views_are_counted_and_empty_files_do_not_stop_scan(self):
        rng = np.random.default_rng(42)
        self.save_image(rng.integers(0, 256, (1024, 1024, 3), dtype=np.uint8), "090.jpg")
        raw = (self.folder / "090.jpg").read_bytes()
        (self.folder / "000.jpg").write_bytes(b"")
        (self.folder / "180.jpg").write_bytes(raw)
        (self.folder / "270.jpg").write_bytes(raw)
        result = self.scan()
        self.assertIsNone(result["incomplete"])
        self.assertEqual(result["corrupt_imgs"], [(1, "000.jpg", "empty file (0 bytes)")])
        self.assertEqual(result["duplicate_views"], [
            (1, "180.jpg", "090.jpg"), (1, "270.jpg", "090.jpg"),
        ])
        self.assertEqual(result["images_ok"], 3)
        self.assertEqual(result["disk_bytes"], 3 * len(raw) + self.meta_path.stat().st_size)

    def test_report_counts_all_warning_categories_separately(self):
        cases = [
            (checker.ScanFindings(unexpected_files=[(1, "extra.jpg")]), checker.DerivedFindings()),
            (checker.ScanFindings(size_outliers=[(1, "000.jpg", 20, "small")]), checker.DerivedFindings()),
            (checker.ScanFindings(), checker.DerivedFindings(orphan_meta={2})),
            (checker.ScanFindings(), checker.DerivedFindings(duplicate_rejected={2: 3})),
            (checker.ScanFindings(), checker.DerivedFindings(never_tried={2})),
        ]
        for findings, derived in cases:
            with self.subTest(findings=findings, derived=derived):
                with contextlib.redirect_stdout(io.StringIO()):
                    failures, warnings = checker._print_report(findings, derived, {1}, True, 0, {}, {1, 2})
                self.assertEqual((failures, warnings), (0, 1))

    def test_cli_clean_warning_and_failure_summaries(self):
        rng = np.random.default_rng(42)
        for name in sorted(checker.EXPECTED_IMAGES):
            self.save_image(rng.integers(0, 256, (1024, 1024, 3), dtype=np.uint8), name)
        (self.root / "completed.csv").write_text("index,panoid\n1,test-pano\n")
        (self.root / "rejects.csv").write_text("index,reason\n")
        flagged = self.root / "flagged.txt"
        flagged.write_text("# Previous review\n")
        command = [sys.executable, "-B", str(Path(checker.__file__).resolve()), str(self.root), "--workers", "1"]
        env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONDONTWRITEBYTECODE="1")
        def run():
            return subprocess.run(command, capture_output=True, text=True, encoding="utf-8", env=env, timeout=30)
        clean = run()
        self.assertEqual(clean.returncode, 0, clean.stderr)
        self.assertIn("All checks passed.", clean.stdout)
        self.assertEqual(flagged.read_text(), "# Previous review\n")
        (self.metadata / "000002.json").write_text(json.dumps(self.base_meta))
        warning = run()
        self.assertEqual(warning.returncode, 0, warning.stderr)
        self.assertIn("No failures; 1 warning(s) found.", warning.stdout)
        self.assertNotIn("All checks passed.", warning.stdout)
        self.assertIn("# 000002", flagged.read_text(encoding="utf-8"))
        (self.folder / "000.jpg").write_bytes(b"")
        failure = run()
        self.assertEqual(failure.returncode, 1, failure.stderr)
        self.assertIn("1 failure(s); 1 warning(s) found.", failure.stdout)

    def test_clean_export_preserves_previous_file(self):
        path = self.root / "flagged.txt"
        path.write_text("000001  # keep old review decision\n")
        old = path.read_bytes()
        self.assertEqual(checker.write_flagged_export(path, []), 0)
        self.assertEqual(path.read_bytes(), old)

    def test_cli_reads_uncompressed_tar_without_extracting(self):
        rng = np.random.default_rng(42)
        for name in sorted(checker.EXPECTED_IMAGES):
            self.save_image(rng.integers(0, 256, (1024, 1024, 3), dtype=np.uint8), name)
        completed = self.root / "completed.csv"
        rejected = self.root / "rejects.csv"
        completed.write_text("index,panoid\n1,test-pano\n")
        rejected.write_text("index,reason\n")

        archive_dir = self.root / "archive-only"
        archive_dir.mkdir()
        archive_path = archive_dir / "salty_data.tar"
        with tarfile.open(archive_path, "w") as archive:
            archive.add(self.root / "images", arcname="salty_data/images")
            archive.add(self.metadata, arcname="salty_data/metadata")
            archive.add(completed, arcname="salty_data/completed.csv")
            archive.add(rejected, arcname="salty_data/rejects.csv")

        command = [
            sys.executable, "-B", str(Path(checker.__file__).resolve()),
            str(archive_path), "--workers", "1", "--no-export",
        ]
        env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONDONTWRITEBYTECODE="1")
        result = subprocess.run(
            command, capture_output=True, text=True, encoding="utf-8", env=env, timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        self.assertIn(f"Archive: {archive_path.resolve()}", result.stdout)
        self.assertIn("Archive root: salty_data", result.stdout)
        self.assertIn("Indexing archive", result.stderr)
        self.assertIn("All checks passed.", result.stdout)
        self.assertEqual(list(archive_dir.iterdir()), [archive_path])

    def test_tar_and_directory_scans_produce_matching_findings(self):
        rng = np.random.default_rng(42)
        for name in sorted(checker.EXPECTED_IMAGES - {"270.jpg"}):
            self.save_image(rng.integers(0, 256, (1024, 1024, 3), dtype=np.uint8), name)
        self.save_image(
            rng.integers(0, 256, (32, 32, 3), dtype=np.uint8), "unexpected.jpg",
        )
        (self.metadata / "000002.json").write_text(json.dumps(self.base_meta))

        archive_path = self.root / "comparison.tar"
        with tarfile.open(archive_path, "w") as archive:
            archive.add(self.metadata, arcname="metadata")
            archive.add(self.root / "images", arcname="images")

        real_tar_open = tarfile.open
        with mock.patch.object(checker.tarfile, "open", wraps=real_tar_open) as archive_open:
            with contextlib.redirect_stderr(io.StringIO()):
                indexed = checker._index_tar_archive(archive_path)
        self.assertEqual(archive_open.call_count, 1)
        directory_findings = checker._scan_all_folders(
            [self.folder], True, self.metadata, None, {1: "test-pano"}, 1,
        )
        archive_findings = checker._scan_archive_folders(
            archive_path, indexed.folders, True, None, {1: "test-pano"}, 1,
        )
        self.assertEqual(indexed.root_name, ".")
        self.assertEqual(indexed.metadata_indices, {1, 2})
        self.assertEqual(archive_findings, directory_findings)

        csv_data = dict(
            completed_set={1}, completed_raw=[1], rejected_set=set(), rejected_raw=[],
        )
        directory_derived = checker._compute_derived(
            directory_findings, csv_data, {1}, True, self.metadata, None,
        )
        archive_derived = checker._compute_derived(
            archive_findings, csv_data, {1}, True, indexed.metadata_indices, None,
        )
        self.assertEqual(archive_derived, directory_derived)
        self.assertEqual(archive_derived.orphan_meta, {2})

    def test_tar_batches_all_folders_in_archive_order(self):
        for idx in range(2, 34):
            (self.root / "images" / f"{idx:06d}").mkdir()
            meta = dict(self.base_meta, index=idx, panoid=f"test-pano-{idx}")
            (self.metadata / f"{idx:06d}.json").write_text(json.dumps(meta))

        archive_path = self.root / "batched.tar"
        with tarfile.open(archive_path, "w") as archive:
            for idx in reversed(range(1, 34)):
                archive.add(
                    self.root / "images" / f"{idx:06d}",
                    arcname=f"salty_data/images/{idx:06d}",
                    recursive=False,
                )
            archive.add(self.metadata, arcname="salty_data/metadata")

        with contextlib.redirect_stderr(io.StringIO()):
            indexed = checker._index_tar_archive(archive_path)
        self.assertIsInstance(indexed.folders, checker.ArchiveFolderIndex)
        self.assertEqual(
            [folder.idx for folder in indexed.folders],
            list(reversed(range(1, 34))),
        )

        findings = checker._scan_archive_folders(
            archive_path, indexed.folders, True, None, {}, 3,
        )
        self.assertEqual(set(findings.empty_folders), set(range(1, 34)))
        self.assertEqual(len(findings.empty_folders), 33)
        self.assertFalse(findings.missing_meta)
        self.assertFalse(findings.corrupt_meta)

    def test_tar_order_matches_first_file_or_empty_directory_header(self):
        archive_path = self.write_archive([
            ("images/000001", None),
            ("images/000002", None),
            ("images/000003", None),
            ("images/000003/000.jpg", b"third-first"),
            ("images/000002/000.jpg", b"second-last"),
        ])
        with contextlib.redirect_stderr(io.StringIO()):
            indexed = checker._index_tar_archive(archive_path)
        self.assertEqual([folder.idx for folder in indexed.folders], [1, 3, 2])

    def test_filesystem_batches_keep_batch_local_csv_checks(self):
        source_coords = {}
        completed_panoids = {}
        for idx in range(1, 18):
            source_coords[idx] = (37.0, -122.0)
            completed_panoids[idx] = "test-pano" if idx == 1 else f"test-pano-{idx}"
            if idx == 1:
                continue
            (self.root / "images" / f"{idx:06d}").mkdir()
            meta = dict(self.base_meta, index=idx, panoid=f"test-pano-{idx}")
            (self.metadata / f"{idx:06d}.json").write_text(json.dumps(meta))

        # Both mismatches are in the second 16-location batch.
        completed_panoids[17] = "different-panoid"
        source_coords[17] = (38.0, -122.0)
        folders, _ = checker._collect_folders(self.root / "images")
        findings = checker._scan_all_folders(
            folders, True, self.metadata, source_coords, completed_panoids, 2,
        )
        self.assertEqual(set(findings.empty_folders), set(range(1, 18)))
        self.assertEqual(findings.panoid_csv_mismatches, [
            (17, "different-panoid", "test-pano-17"),
        ])
        self.assertEqual(findings.coord_mismatches, [
            (17, 37.0, -122.0, 38.0, -122.0),
        ])

    def test_tar_root_selection_prefers_score_then_depth_then_image_order(self):
        cases = [
            (
                [
                    ("small/images/000001", None),
                    ("large/nested/images/000002/000.jpg", b"first"),
                    ("large/nested/images/000002/090.jpg", b"second"),
                ],
                "large/nested", {2},
            ),
            (
                [("deep/root/images/000001", None), ("shallow/images/000002", None)],
                "shallow", {2},
            ),
            (
                [
                    ("second/completed.csv", b"index,panoid\n2,second\n"),
                    ("first/images/000001", None),
                    ("second/images/000002", None),
                ],
                "first", {1},
            ),
        ]
        for members, expected_root, expected_indices in cases:
            with self.subTest(root=expected_root):
                archive_path = self.write_archive(members)
                with contextlib.redirect_stderr(io.StringIO()):
                    indexed = checker._index_tar_archive(archive_path)
                self.assertEqual(indexed.root_name, expected_root)
                self.assertEqual(indexed.disk_indices, expected_indices)

    def test_tar_duplicate_members_and_metadata_names_are_preserved(self):
        metadata = self.meta_path.read_bytes()
        archive_path = self.write_archive([
            ("metadata/000001.json", b"invalid old metadata"),
            ("images/000001/000.jpg", b"first image"),
            ("images/000001/000.jpg", b"second image"),
            ("images/000001/note.txt", b"extra file"),
            ("images/000001/nested/ignored.jpg", b"nested file"),
            ("../images/000003/000.jpg", b"unsafe path"),
            ("metadata/000001.json", metadata),
            ("metadata/1.json", b"noncanonical metadata"),
            ("metadata/000002.json", b"orphan metadata"),
        ])
        with contextlib.redirect_stderr(io.StringIO()):
            indexed = checker._index_tar_archive(archive_path)
        self.assertEqual(indexed.disk_indices, {1})
        self.assertEqual(indexed.metadata_indices, {1, 2})
        folder, = indexed.folders
        self.assertEqual([member.name for member in folder.images], [
            "000.jpg", "000.jpg", "note.txt",
        ])
        self.assertEqual(
            [checker._read_archive_member(archive_path, member) for member in folder.images],
            [b"first image", b"second image", b"extra file"],
        )
        checker._init_worker(None, {}, archive_path)
        result = checker._process_archive_folder(folder, True)
        self.assertIsNone(result["corrupt_meta"])
        self.assertEqual(result["panoid"], "test-pano")
        self.assertEqual(result["unexpected_files"], [(1, "note.txt")])

    def test_archive_readers_reject_short_reads(self):
        archive_path = self.root / "truncated.tar"
        archive_path.write_bytes(b"short")
        member = checker.ArchiveMember("000.jpg", 0, 10)
        with self.assertRaisesRegex(OSError, "short read: expected 10 bytes, got 5"):
            checker._read_archive_member(archive_path, member)

        checker._init_worker(None, {}, archive_path)
        folder = checker.ArchiveFolder(1, (member,))
        result = checker._process_archive_folder(folder, False)
        self.assertEqual(result["corrupt_imgs"], [
            (1, "000.jpg", "cannot read: short read: expected 10 bytes, got 5"),
        ])
        self.assertEqual(result["disk_bytes"], 0)


if __name__ == "__main__":
    unittest.main()
