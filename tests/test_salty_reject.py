import contextlib
import io
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

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
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(self.snapshot(), original)

        result = self.run_cli('--reject-all-flagged', answer='yes\n')
        self.assertEqual(result.returncode, 0, result.stderr)
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
        bookkeeping = csv_names | {'rejects.csv', 'rejects_100.csv',
                                  'rejected_archive/records/000001.json',
                                  'rejected_archive/records/000002.json'}
        self.assertEqual({name: data for name, data in after.items() if name not in bookkeeping}, expected)
        self.assertEqual(set(after) - set(expected), bookkeeping)
        self.assertEqual(pd.read_csv(first_csv)['index'].tolist(), [3, 10])
        self.assertEqual(pd.read_csv(second_csv)['index'].tolist(), [4, 123])
        self.assertEqual(pd.read_csv(self.root / 'rejects.csv')['index'].tolist(), [1])
        self.assertEqual(pd.read_csv(self.root / 'rejects_100.csv')['index'].tolist(), [2])
        result = self.run_cli('--reject-all-flagged', answer='yes\n')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('2 skipped', result.stdout)
        self.assertEqual(self.snapshot(), after)

        undo = self.root / 'undo_list.txt'
        undo.write_text('1\n2\n')
        before_undo = self.snapshot()
        result = self.run_cli('--undo', '--from-file', str(undo), '--dry-run')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.snapshot(), before_undo)
        result = self.run_cli('--undo', '--from-file', str(undo), answer='yes\n')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('2 restored', result.stdout)
        restored = self.snapshot()
        extras = {'undo_list.txt', 'rejects.csv', 'rejects_100.csv'}
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
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.snapshot(), restored)

    def test_incompatible_modes_are_rejected(self):
        for mode in ('--undo', '--purge'):
            with self.subTest(mode=mode):
                result = self.run_cli('--reject-all-flagged', mode)
                self.assertEqual(result.returncode, 2)
                self.assertIn('cannot be used', result.stderr)


if __name__ == '__main__':
    unittest.main()
