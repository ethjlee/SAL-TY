import contextlib
import io
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
from aiohttp import ClientConnectionError, ConnectionTimeoutError

import salty_image_grabber as scraper


class LifecycleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.output = Path(self.temp.name)
        self.paths = scraper._Paths.from_output_dir(self.output)
        self.coords = pd.DataFrame({"index": [101, 102], "lat": [1., 2.], "lon": [3., 4.]})
        self.args = SimpleNamespace(start_index=0, end_index=None, backfill_only=False)

    def run_main(self, result=(True, None)):
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(scraper, "OUTPUT_DIR", self.output))
            stack.enter_context(patch.object(scraper, "COORDS_FILE", str(self.output / "coords.csv")))
            self.coords.to_csv(self.output / "coords.csv", index=False)
            stack.enter_context(patch.object(scraper, "parse_args", return_value=self.args))
            stack.enter_context(patch.object(scraper, "setup_logging"))
            stack.enter_context(patch.object(scraper.time, "sleep"))
            backfill = stack.enter_context(patch.object(scraper, "backfill_metadata", return_value=True))
            download = stack.enter_context(patch.object(scraper, "download_location"))
            if isinstance(result, list) or callable(result):
                download.side_effect = result
            else:
                download.return_value = result
            output = stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            stack.enter_context(contextlib.redirect_stderr(io.StringIO()))
            code = scraper.main()
            return code, download, backfill, output.getvalue()

    def test_finished_batch_skips_download_and_backfill(self):
        pd.DataFrame({"index": [101, 102, 999]}).to_csv(self.paths.completed_csv, index=False)
        code, download, backfill, output = self.run_main()
        self.assertEqual(code, 0)
        download.assert_not_called()
        backfill.assert_not_called()
        self.assertIn("All locations processed!", output)

    def test_successful_batch_exits_zero_without_backfill(self):
        code, download, backfill, output = self.run_main()
        self.assertEqual(code, 0)
        self.assertEqual(download.call_count, 2)
        backfill.assert_not_called()
        self.assertIn("DOWNLOAD COMPLETE", output)

    def test_isolated_error_counts_as_attempted_and_batch_finishes(self):
        code, _, _, output = self.run_main([(False, "exception"), (True, None)])
        self.assertEqual(code, 0)
        self.assertNotIn("BATCH INCOMPLETE", output)
        self.assertIn("DOWNLOAD COMPLETE", output)

    def test_consecutive_timeouts_exit_failure_at_threshold(self):
        self.coords = pd.concat([self.coords] * 5, ignore_index=True)
        self.coords["index"] = range(10)
        code, download, _, output = self.run_main((False, "timeout"))
        self.assertEqual(code, 1)
        self.assertEqual(download.call_count, scraper.MAX_CONSECUTIVE_TIMEOUTS)
        self.assertNotIn("DOWNLOAD COMPLETE", output)

    def test_consecutive_generic_errors_exit_failure(self):
        self.coords = pd.concat([self.coords] * 6, ignore_index=True)
        self.coords["index"] = range(12)
        code, download, _, _ = self.run_main((False, "exception"))
        self.assertEqual(code, 1)
        self.assertEqual(download.call_count, scraper.MAX_CONSECUTIVE_ERRORS)

    def test_permanent_rejects_can_finish_batch(self):
        code, _, _, _ = self.run_main([(False, "no_panorama"), (False, "qc_failed_non_google_copyright")])
        self.assertEqual(code, 0)

    def test_all_old_rejects_are_skipped_regardless_of_reason(self):
        pd.DataFrame({"index": [101, 102], "reason": ["error_ConnectionTimeoutError", "no_panorama_found"]}).to_csv(self.paths.rejects_csv, index=False)
        code, download, _, _ = self.run_main()
        self.assertEqual(code, 0)
        download.assert_not_called()

    def test_deleting_rejects_retries_only_locations_not_completed(self):
        pd.DataFrame({"index": [101, 102], "reason": ["error_ValueError", "network_timeout"]}).to_csv(self.paths.rejects_csv, index=False)
        pd.DataFrame({"index": [102]}).to_csv(self.paths.completed_csv, index=False)
        code, download, _, _ = self.run_main()
        self.assertEqual(code, 0)
        download.assert_not_called()
        self.paths.rejects_csv.unlink()
        code, download, _, _ = self.run_main()
        self.assertEqual(code, 0)
        self.assertEqual(download.call_count, 1)
        self.assertEqual(int(download.call_args.args[0].iloc[0]), 101)

    def test_threshold_on_final_location_stays_stopped(self):
        self.coords = pd.DataFrame({"index": range(5), "lat": [1.] * 5, "lon": [2.] * 5})
        code, download, _, output = self.run_main((False, "timeout"))
        self.assertEqual(code, 0)
        self.assertEqual(download.call_count, 5)
        self.assertIn("DOWNLOAD COMPLETE", output)

    def test_restart_resumes_after_recorded_errors_without_retrying_them(self):
        self.coords = pd.DataFrame({"index": range(7), "lat": [1.] * 7, "lon": [2.] * 7})
        with patch.object(scraper.streetview, "find_panorama", side_effect=ConnectionTimeoutError("offline")):
            code, download, _, _ = self.run_main(scraper.download_location)
            self.assertEqual(code, 1)
            self.assertEqual(download.call_count, 5)
            self.assertEqual(scraper._load_index_set(self.paths.rejects_csv), set(range(5)))
            code, download, _, _ = self.run_main(scraper.download_location)
            self.assertEqual(code, 0)
            self.assertEqual(download.call_count, 2)
            self.assertEqual(scraper._load_index_set(self.paths.rejects_csv), set(range(7)))
            code, download, _, _ = self.run_main(scraper.download_location)
            self.assertEqual(code, 0)
            download.assert_not_called()

    def test_explicit_backfill_does_not_scrape(self):
        self.args.backfill_only = True
        code, download, backfill, _ = self.run_main()
        self.assertEqual(code, 0)
        download.assert_not_called()
        backfill.assert_called_once()

    def test_invalid_progress_is_not_silently_ignored(self):
        self.paths.completed_csv.write_text("wrong_column\n101\n")
        with self.assertRaises(KeyError):
            scraper._load_index_set(self.paths.completed_csv)

    def test_aiohttp_network_errors_are_recorded_as_rejects(self):
        for error in (ConnectionTimeoutError("slow"), ClientConnectionError("offline"), TimeoutError("slow")):
            with self.subTest(error=type(error).__name__), patch.object(scraper.streetview, "find_panorama", side_effect=error):
                self.assertEqual(scraper.download_location(self.coords.iloc[0], set(), set(), self.paths), (False, "timeout"))
                self.assertEqual(scraper._load_index_set(self.paths.rejects_csv), {101})
                self.assertEqual(pd.read_csv(self.paths.rejects_csv).iloc[-1]["reason"], "network_timeout")

    def test_generic_errors_are_recorded_as_rejects(self):
        with patch.object(scraper.streetview, "find_panorama", side_effect=ValueError("bad response")):
            self.assertEqual(scraper.download_location(self.coords.iloc[0], set(), set(), self.paths), (False, "exception"))
        self.assertEqual(scraper._load_index_set(self.paths.rejects_csv), {101})
        self.assertEqual(pd.read_csv(self.paths.rejects_csv).iloc[0]["reason"], "error_ValueError")

    def test_extraction_failures_are_recorded_as_rejects(self):
        pano = SimpleNamespace(id="test-pano", copyright_message="Google")
        with patch.object(scraper.streetview, "find_panorama", return_value=pano), patch.object(scraper.streetview, "download_panorama"), patch.object(scraper.Image, "open"), patch.object(scraper, "extract_perspective_views", return_value=False):
            self.assertEqual(scraper.download_location(self.coords.iloc[0], set(), set(), self.paths), (False, "extraction_failed"))
        self.assertEqual(pd.read_csv(self.paths.rejects_csv).iloc[0]["reason"], "view_extraction_failed")

    def test_backfill_only_touches_assigned_metadata(self):
        scraper.setup_directories(self.paths)
        complete = dict.fromkeys(["heading", "pitch", "roll", "street_names", "address", "country_code", "headings", "view_resolution", "view_fov"])
        (self.paths.metadata / "000101.json").write_text(scraper.json.dumps(complete))
        other = self.paths.metadata / "000999.json"
        other.write_text("invalid json from another batch")
        with patch.object(scraper.streetview, "find_panorama") as api:
            self.assertTrue(scraper.backfill_metadata(self.paths, self.coords))
        api.assert_not_called()
        self.assertEqual(other.read_text(), "invalid json from another batch")

    def test_watchdog_terminates_stalled_process_with_failure(self):
        result = subprocess.run([sys.executable, "-c", "import time; from salty_image_grabber import ProgressWatchdog\nwith ProgressWatchdog(0.15): time.sleep(10)"], capture_output=True, timeout=30)
        self.assertEqual(result.returncode, 1)
        self.assertIn(b"scraper stalled", result.stderr)


if __name__ == "__main__":
    unittest.main()
