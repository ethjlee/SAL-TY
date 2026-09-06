import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

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


if __name__ == "__main__":
    unittest.main()
