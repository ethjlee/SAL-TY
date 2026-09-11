import contextlib
import io
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

import salty_reject as reject
from salty_check import write_flagged_export


class RejectTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.flags = self.root / 'flagged.txt'
        write_flagged_export(self.flags, [('blurry_imgs', [(1, 'low score'), (1, 'duplicate')])])
        with self.flags.open('a', encoding='utf-8') as stream:
            stream.write('\n# A note\n# 123 locations checked\n#\n  000002 # manual reason\n3\n')

    def run_cli(self, *args, answer=None):
        return subprocess.run(
            [sys.executable, str(Path(reject.__file__)), str(self.root), *args],
            input=answer, text=True, capture_output=True,
        )

    def test_default_only_reads_uncommented_entries(self):
        self.assertEqual(reject.parse_index_file(self.flags), [(2, 'manual reason'), (3, 'manual_reject')])

    def test_commented_only_treats_removed_markers_as_keep_decisions(self):
        entries = reject.parse_index_file(self.flags, commented_only=True)
        self.assertEqual(entries, [(1, 'blurry_imgs: low score')])

    def test_default_flagged_file_rejects_comments_and_keeps_uncommented(self):
        csv = self.root / 'completed.csv'
        csv.write_text('index,lat,lon,panoid\n1,10,20,pano1\n2,11,21,pano2\n')
        self.flags.write_text('# 1 # reject this\n2 # keep this\n')

        result = self.run_cli('--dry-run')

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('Indices : 1', result.stdout)
        self.assertIn('[DRY RUN] 000001', result.stdout)
        self.assertNotIn('[DRY RUN] 000002', result.stdout)

    def test_explicit_flagged_file_uses_same_review_convention(self):
        csv = self.root / 'completed.csv'
        csv.write_text('index,lat,lon,panoid\n1,10,20,pano1\n2,11,21,pano2\n')
        self.flags.write_text('# 1 # reject this\n2 # keep this\n')

        result = self.run_cli('--from-file', str(self.flags), '--dry-run')

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('Indices : 1', result.stdout)
        self.assertIn('[DRY RUN] 000001', result.stdout)
        self.assertNotIn('[DRY RUN] 000002', result.stdout)

    def test_all_flags_preserves_reasons_and_deduplicates(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            entries = reject.parse_index_file(self.flags, include_commented=True)
        self.assertEqual(entries, [(1, 'blurry_imgs: low score'), (2, 'manual reason'), (3, 'manual_reject')])
        self.assertEqual(output.getvalue(), '')

    def test_cli_dry_run_then_reject_and_undo(self):
        csv = self.root / 'completed.csv'
        csv.write_text('index,lat,lon,panoid\n1,10,20,pano1\n2,11,21,pano2\n3,12,22,pano3\n4,13,23,pano4\n')
        original = csv.read_bytes()
        flags_original = self.flags.read_bytes()
        result = self.run_cli('--reject-all-flagged', '--dry-run')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('3 to reject', result.stdout)
        self.assertEqual(csv.read_bytes(), original)
        self.assertFalse((self.root / 'rejected_archive').exists())
        result = self.run_cli('--reject-all-flagged', answer='yes\n')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('3 rejected', result.stdout)
        self.assertEqual(pd.read_csv(csv)['index'].tolist(), [4])
        rejected = pd.read_csv(self.root / 'rejects.csv')
        self.assertEqual(rejected['index'].tolist(), [1, 2, 3])
        self.assertEqual(rejected.iloc[0]['reason'], 'blurry_imgs: low score')
        self.assertEqual(self.flags.read_bytes(), flags_original)
        undo = self.root / 'undo.txt'
        undo.write_text('1\n2\n3\n')
        result = self.run_cli('--undo', '--from-file', str(undo), answer='yes\n')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('3 restored', result.stdout)
        self.assertEqual(set(pd.read_csv(csv)['index']), {1, 2, 3, 4})
        self.assertEqual((self.root / 'reject_list.txt').read_text(), '')

    def test_default_undo_accumulates_and_restores_separate_reject_runs(self):
        csv = self.root / 'completed.csv'
        csv.write_text('index,lat,lon,panoid\n1,10,20,pano1\n2,11,21,pano2\n3,12,22,pano3\n')

        self.flags.write_text('# 1 # first batch\n')
        first = self.run_cli(answer='yes\n')
        self.assertEqual(first.returncode, 0, first.stderr)
        self.assertEqual((self.root / 'reject_list.txt').read_text(), '1\n')

        self.flags.write_text('# 2 # second batch\n')
        second = self.run_cli(answer='yes\n')
        self.assertEqual(second.returncode, 0, second.stderr)
        self.assertEqual((self.root / 'reject_list.txt').read_text(), '1\n2\n')
        self.assertEqual(pd.read_csv(csv)['index'].tolist(), [3])

        undone = self.run_cli('--undo', answer='yes\n')
        self.assertEqual(undone.returncode, 0, undone.stderr)
        self.assertIn('2 restored', undone.stdout)
        self.assertEqual(set(pd.read_csv(csv)['index']), {1, 2, 3})
        self.assertEqual((self.root / 'reject_list.txt').read_text(), '')

    def test_subset_undo_keeps_other_entries_in_automatic_list(self):
        csv = self.root / 'completed.csv'
        csv.write_text('index,lat,lon,panoid\n1,10,20,pano1\n2,11,21,pano2\n')
        self.flags.write_text('# 1\n# 2\n')
        rejected = self.run_cli(answer='yes\n')
        self.assertEqual(rejected.returncode, 0, rejected.stderr)

        subset = self.root / 'undo_subset.txt'
        subset.write_text('1\n')
        restored = self.run_cli('--undo', '--from-file', str(subset), answer='yes\n')
        self.assertEqual(restored.returncode, 0, restored.stderr)
        self.assertEqual((self.root / 'reject_list.txt').read_text(), '2\n')
        self.assertEqual(pd.read_csv(csv)['index'].tolist(), [1])

    def test_purge_clears_automatic_reject_list(self):
        csv = self.root / 'completed.csv'
        csv.write_text('index,lat,lon,panoid\n1,10,20,pano1\n')
        self.flags.write_text('# 1\n')
        rejected = self.run_cli(answer='yes\n')
        self.assertEqual(rejected.returncode, 0, rejected.stderr)
        self.assertEqual((self.root / 'reject_list.txt').read_text(), '1\n')

        purged = self.run_cli('--purge', answer='PURGE\n')
        self.assertEqual(purged.returncode, 0, purged.stderr)
        self.assertFalse((self.root / 'rejected_archive').exists())
        self.assertEqual((self.root / 'reject_list.txt').read_text(), '')

    def test_reject_list_write_failure_preserves_recovery_record(self):
        paths = reject._Paths.from_data_dir(self.root)
        paths.arch_records.mkdir(parents=True)
        record = paths.arch_records / '000001.json'
        record.write_text('{}')
        output = io.StringIO()

        with mock.patch.object(Path, 'write_text', side_effect=OSError('read-only')):
            with contextlib.redirect_stdout(output):
                updated = reject._refresh_reject_list(paths)

        self.assertFalse(updated)
        self.assertTrue(record.exists())
        self.assertIn('Could not update reject_list.txt', output.getvalue())

    def test_result_codes_distinguish_benign_skips_from_unfulfilled_entries(self):
        (self.root / 'rejects.csv').write_text('index,reason\n1,old\n')
        already_done = reject.do_reject(
            self.root, [(1, 'already done')], dry_run=True,
        )
        self.assertEqual(already_done.skipped, 1)
        self.assertEqual(already_done.exit_code, 0)

        records = self.root / 'rejected_archive' / 'records'
        records.mkdir(parents=True)
        (records / '000002.json').write_text('{}')

        result = reject.do_reject(
            self.root,
            [(1, 'already done'), (2, 'blocked'), (3, 'missing')],
            dry_run=True,
        )

        self.assertEqual(result.skipped, 1)
        self.assertEqual(result.blocked, 1)
        self.assertEqual(result.not_found, 1)
        self.assertEqual(result.exit_code, 1)

    def test_unreadable_progress_csv_aborts_before_mutation(self):
        completed = self.root / 'completed.csv'
        completed.write_text('wrong_column\n1\n')

        result = reject.do_reject(self.root, [(1, 'test')], dry_run=False)

        self.assertEqual(result.exit_code, 1)
        self.assertEqual(result.errors, 1)
        self.assertEqual(completed.read_text(), 'wrong_column\n1\n')
        self.assertFalse((self.root / 'rejected_archive').exists())

    def test_recovery_record_write_failure_is_reported_without_mutation(self):
        completed = self.root / 'completed.csv'
        completed.write_text('index,lat,lon,panoid\n1,10,20,pano1\n')

        with mock.patch.object(reject.json, 'dumps', side_effect=TypeError('bad record')):
            result = reject.do_reject(self.root, [(1, 'test')], dry_run=False)

        self.assertEqual(result.exit_code, 1)
        self.assertEqual(result.errors, 1)
        self.assertEqual(pd.read_csv(completed)['index'].tolist(), [1])
        self.assertFalse((self.root / 'rejects.csv').exists())

    def test_reject_continues_after_one_entry_conflicts(self):
        completed = self.root / 'completed.csv'
        completed.write_text(
            'index,lat,lon,panoid\n1,10,20,pano1\n2,11,21,pano2\n'
        )
        live = self.root / 'images' / '000001'
        live.mkdir(parents=True)
        (live / 'live.jpg').write_bytes(b'live')
        archived = self.root / 'rejected_archive' / 'images' / '000001'
        archived.mkdir(parents=True)
        (archived / 'old.jpg').write_bytes(b'old')

        result = reject.do_reject(
            self.root, [(1, 'conflict'), (2, 'works')], dry_run=False,
        )

        self.assertEqual(result.exit_code, 1)
        self.assertEqual(result.succeeded, 1)
        self.assertEqual(result.errors, 1)
        self.assertEqual(pd.read_csv(completed)['index'].tolist(), [1])
        self.assertEqual(pd.read_csv(self.root / 'rejects.csv')['index'].tolist(), [2])

    def test_reject_list_refresh_failure_sets_failure_status(self):
        completed = self.root / 'completed.csv'
        completed.write_text('index,lat,lon,panoid\n1,10,20,pano1\n')

        with mock.patch.object(reject, '_refresh_reject_list', return_value=False):
            result = reject.do_reject(self.root, [(1, 'test')], dry_run=False)

        self.assertEqual(result.succeeded, 1)
        self.assertEqual(result.errors, 1)
        self.assertEqual(result.exit_code, 1)

    def test_unreadable_recovery_record_sets_failure_status(self):
        records = self.root / 'rejected_archive' / 'records'
        records.mkdir(parents=True)
        (records / '000001.json').write_text('{')

        result = reject.do_undo(self.root, [(1, '')], dry_run=False)

        self.assertEqual(result.succeeded, 0)
        self.assertEqual(result.errors, 1)
        self.assertEqual(result.exit_code, 1)

    def test_purge_failure_sets_failure_status(self):
        archive = self.root / 'rejected_archive'
        archive.mkdir()

        with mock.patch('builtins.input', return_value='PURGE'):
            with mock.patch.object(reject.shutil, 'rmtree', side_effect=OSError('busy')):
                result = reject.do_purge(self.root, dry_run=False)

        self.assertEqual(result.errors, 1)
        self.assertEqual(result.exit_code, 1)
        self.assertTrue(archive.exists())

    def test_archive_destination_conflict_aborts_entry_without_mutation(self):
        csv = self.root / 'completed.csv'
        csv.write_text('index,lat,lon,panoid\n1,10,20,pano1\n')
        live = self.root / 'images' / '000001'
        live.mkdir(parents=True)
        (live / 'live.jpg').write_bytes(b'live')
        archived = self.root / 'rejected_archive' / 'images' / '000001'
        archived.mkdir(parents=True)
        (archived / 'old.jpg').write_bytes(b'old')
        self.flags.write_text('# 1\n')

        result = self.run_cli(answer='yes\n')

        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn('archive destination', result.stdout)
        self.assertEqual(pd.read_csv(csv)['index'].tolist(), [1])
        self.assertTrue((live / 'live.jpg').exists())
        self.assertFalse((self.root / 'rejected_archive' / 'records' / '000001.json').exists())
        self.assertFalse((self.root / 'rejects.csv').exists())

    def test_reject_csv_removal_failure_does_not_add_reject(self):
        csv = self.root / 'completed.csv'
        csv.write_text('index,lat,lon,panoid\n1,10,20,pano1\n')

        with mock.patch.object(reject, '_batch_remove_from_csv', return_value=None):
            result = reject.do_reject(self.root, [(1, 'test')], dry_run=False)

        self.assertEqual(result.exit_code, 1)
        self.assertEqual(result.errors, 1)
        self.assertEqual(pd.read_csv(csv)['index'].tolist(), [1])
        self.assertFalse((self.root / 'rejects.csv').exists())
        self.assertTrue((self.root / 'rejected_archive' / 'records' / '000001.json').exists())

    def test_failed_rejects_cleanup_preserves_record_and_retry_is_idempotent(self):
        csv = self.root / 'completed.csv'
        csv.write_text('index,lat,lon,panoid\n1,10,20,pano1\n')
        reject.do_reject(self.root, [(1, 'test')], dry_run=False)
        record = self.root / 'rejected_archive' / 'records' / '000001.json'

        with mock.patch.object(reject, '_batch_remove_from_csv', return_value=None):
            result = reject.do_undo(self.root, [(1, '')], dry_run=False)

        self.assertEqual(result.exit_code, 1)
        self.assertTrue(record.exists())
        self.assertEqual(pd.read_csv(csv)['index'].tolist(), [1])
        self.assertEqual(pd.read_csv(self.root / 'rejects.csv')['index'].tolist(), [1])

        reject.do_undo(self.root, [(1, '')], dry_run=False)
        self.assertFalse(record.exists())
        self.assertEqual(pd.read_csv(csv)['index'].tolist(), [1])
        self.assertTrue(pd.read_csv(self.root / 'rejects.csv').empty)

    def test_partial_restore_does_not_append_completed_row_and_can_retry(self):
        csv = self.root / 'completed.csv'
        csv.write_text('index,lat,lon,panoid\n1,10,20,pano1\n')
        image = self.root / 'images' / '000001' / 'view.jpg'
        image.parent.mkdir(parents=True)
        image.write_bytes(b'image')
        metadata = self.root / 'metadata' / '000001.json'
        metadata.parent.mkdir(parents=True)
        metadata.write_text('{"original_lat":10,"original_lon":20,"panoid":"pano1"}')
        reject.do_reject(self.root, [(1, 'test')], dry_run=False)

        metadata.write_text('conflict')
        result = reject.do_undo(self.root, [(1, '')], dry_run=False)

        self.assertEqual(result.exit_code, 1)
        self.assertEqual(result.partial, 1)
        record = self.root / 'rejected_archive' / 'records' / '000001.json'
        self.assertTrue(record.exists())
        self.assertTrue(pd.read_csv(csv).empty)

        metadata.unlink()
        reject.do_undo(self.root, [(1, '')], dry_run=False)
        self.assertFalse(record.exists())
        self.assertEqual(pd.read_csv(csv)['index'].tolist(), [1])

    def test_bare_undo_supports_legacy_records_without_a_reject_list(self):
        (self.root / 'completed.csv').write_text('index,lat,lon,panoid\n')
        (self.root / 'rejects.csv').write_text(
            'timestamp,index,lat,lon,reason,panoid\nnow,1,10,20,old,pano1\n'
        )
        records = self.root / 'rejected_archive' / 'records'
        records.mkdir(parents=True)
        (records / '000001.json').write_text(
            '{"completed_row":{"index":1,"lat":10,"lon":20,"panoid":"pano1"},'
            '"completed_source_file":"completed.csv","had_completed_row":true,'
            '"had_images":false,"had_metadata":false}'
        )

        restored = self.run_cli('--undo', answer='yes\n')
        self.assertEqual(restored.returncode, 0, restored.stderr)
        self.assertIn('using archived recovery records', restored.stdout)
        self.assertEqual(pd.read_csv(self.root / 'completed.csv')['index'].tolist(), [1])
        self.assertEqual((self.root / 'reject_list.txt').read_text(), '')

    def snapshot(self):
        return {
            path.relative_to(self.root).as_posix(): path.read_bytes()
            for path in self.root.rglob('*') if path.is_file()
        }

    def test_files_are_scoped_to_listed_locations_and_undo_restores_bytes(self):
        # 1 is commented, 2 is uncommented; all other locations must survive.
        self.flags.write_text(
            '# SALTY flagged locations\n# --- blurry_imgs (2) ---\n'
            '# 000001 # blurry_imgs: 000.jpg: low score\n'
            '# 000001 # duplicate flag\n000002 # manual reason\n'
            '# 999999 # nonexistent location\n# 123 locations checked\n'
            '# ../000004 # invalid path\n# 000004.json # not an index\n'
        )
        for idx in (1, 2, 3, 4, 10, 123):
            folder = self.root / 'images' / f'{idx:06d}'
            folder.mkdir(parents=True)
            for name in ('000.jpg', '090.jpg', 'nested/extra.bin'):
                image = folder / name
                image.parent.mkdir(exist_ok=True)
                image.write_bytes(f'{idx}/{name}'.encode() + bytes(range(256)))
            meta = self.root / 'metadata' / f'{idx:06d}.json'
            meta.parent.mkdir(exist_ok=True)
            meta.write_text('{"original_lat":10,"original_lon":20,"panoid":"pano"}')
        for name in ('images/000001_extra/keep.jpg', 'metadata/000001.json.bak', 'notes.txt'):
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'leave this alone')
        first_csv = self.root / 'completed.csv'
        second_csv = self.root / 'completed_100.csv'
        first_csv.write_text('index,lat,lon,panoid\n1,10,20,a\n3,11,21,c\n10,12,22,j\n')
        second_csv.write_text('index,lat,lon,panoid\n2,10,20,b\n4,11,21,d\n123,12,22,k\n')
        original_frames = {p: pd.read_csv(p) for p in (first_csv, second_csv)}
        original = self.snapshot()
        for args, answer in ((('--reject-all-flagged', '--dry-run'), None),
                             (('--reject-all-flagged',), 'no\n')):
            result = self.run_cli(*args, answer=answer)
            expected_code = 1 if '--dry-run' in args else 0
            self.assertEqual(result.returncode, expected_code, result.stderr)
            self.assertEqual(self.snapshot(), original)

        result = self.run_cli('--reject-all-flagged', answer='yes\n')
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn('2 rejected', result.stdout)
        self.assertIn('1 not found', result.stdout)
        after = self.snapshot()
        moved = {name for name in original if
                 name.startswith(('images/000001/', 'images/000002/')) or
                 name in ('metadata/000001.json', 'metadata/000002.json')}
        csv_names = {'completed.csv', 'completed_100.csv'}
        expected = {name: data for name, data in original.items()
                    if name not in moved and name not in csv_names}
        expected.update({'rejected_archive/' + name: original[name] for name in moved})
        bookkeeping = csv_names | {'rejects.csv', 'rejects_100.csv', 'reject_list.txt',
                                  'rejected_archive/records/000001.json',
                                  'rejected_archive/records/000002.json'}
        self.assertEqual({name: data for name, data in after.items() if name not in bookkeeping}, expected)
        self.assertEqual(set(after) - set(expected), bookkeeping)
        self.assertEqual(pd.read_csv(first_csv)['index'].tolist(), [3, 10])
        self.assertEqual(pd.read_csv(second_csv)['index'].tolist(), [4, 123])
        self.assertEqual(pd.read_csv(self.root / 'rejects.csv')['index'].tolist(), [1])
        self.assertEqual(pd.read_csv(self.root / 'rejects_100.csv')['index'].tolist(), [2])
        result = self.run_cli('--reject-all-flagged', answer='yes\n')
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn('2 skipped', result.stdout)
        self.assertEqual(self.snapshot(), after)

        undo = self.root / 'undo_subset.txt'
        undo.write_text('1\n2\n')
        before_undo = self.snapshot()
        result = self.run_cli('--undo', '--from-file', str(undo), '--dry-run')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.snapshot(), before_undo)
        result = self.run_cli('--undo', '--from-file', str(undo), answer='yes\n')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('2 restored', result.stdout)
        restored = self.snapshot()
        extras = {'reject_list.txt', 'undo_subset.txt', 'rejects.csv', 'rejects_100.csv'}
        self.assertEqual({name: data for name, data in restored.items()
                          if name not in csv_names | extras},
                         {name: data for name, data in original.items() if name not in csv_names})
        for path, frame in original_frames.items():
            pd.testing.assert_frame_equal(
                pd.read_csv(path).sort_values('index').reset_index(drop=True), frame,
            )
        for name in ('rejects.csv', 'rejects_100.csv'):
            self.assertTrue(pd.read_csv(self.root / name).empty)
        result = self.run_cli('--undo', '--from-file', str(undo), answer='yes\n')
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertEqual(self.snapshot(), restored)

    def test_incompatible_modes_are_rejected(self):
        for mode in ('--undo', '--purge'):
            with self.subTest(mode=mode):
                result = self.run_cli('--reject-all-flagged', mode)
                self.assertEqual(result.returncode, 2)
                self.assertIn('cannot be used', result.stderr)

        result = self.run_cli('--undo', '--purge')
        self.assertEqual(result.returncode, 2)
        self.assertIn('not allowed with argument', result.stderr)


if __name__ == '__main__':
    unittest.main()
