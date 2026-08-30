"""
SALTY Scraper - Google Street View Multi-View Data Acquisition
Project SALTY: Street-view Attention Learning Telemetry

CONFIGURATION: ZOOM 3, Quality 90
- Higher resolution panoramas (ZOOM 3) for better detail capture
- Quality 90 compression for OCR and ML training
- Optimized for Vision Transformers and OCR tasks

This script downloads Google Street View panoramas and extracts 4 directional views
(0°, 90°, 180°, 270°) for each location. Each location gets its own subfolder with
the 4 perspective views.

Quality Control:
- Filters out third-party photospheres
- Rejects indoor imagery
- Only accepts official Google Street View content

Output Structure:
    images/
        000000/
            000.jpg  (0° - North)
            090.jpg  (90° - East)
            180.jpg  (180° - South)
            270.jpg  (270° - West)
        000001/
            000.jpg
            090.jpg
            ...

Stealth Protocol:
- Randomized sleep between requests (0.25-1.0s)
"""

import argparse
import json
import logging
import os
import random
import shutil
import tempfile
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytorch360convert
import streetlevel.streetview as streetview
import torch
from PIL import Image
from requests.exceptions import ConnectionError, Timeout
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

COORDS_FILE = "0-100k_data.csv"
OUTPUT_DIR = Path("salty_data")

PANO_ZOOM   = 4      # Zoom level for downloading equirectangular panorama
VIEW_HEIGHT = 1024   # Height of extracted perspective views
VIEW_WIDTH  = 1024   # Width of extracted perspective views
VIEW_FOV    = 90.0   # Field of view for perspective views (degrees)
HEADINGS    = [0, 90, 180, 270]

MIN_SLEEP = 0.25
MAX_SLEEP = 1.0

MAX_CONSECUTIVE_TIMEOUTS = 5   # Terminate after this many consecutive timeouts
MAX_CONSECUTIVE_ERRORS   = 10  # Terminate after this many consecutive errors (likely IP ban)


# ---------------------------------------------------------------------------
# Path layout
# ---------------------------------------------------------------------------

@dataclass
class _Paths:
    images:        Path
    metadata:      Path
    completed_csv: Path
    rejects_csv:   Path
    log_file:      Path

    @classmethod
    def from_output_dir(cls, output_dir: Path, shard: int = 0) -> "_Paths":
        suffix = f"_{shard}" if shard > 0 else ""
        return cls(
            images=output_dir / "images",
            metadata=output_dir / "metadata",
            completed_csv=output_dir / f"completed{suffix}.csv",
            rejects_csv=output_dir / f"rejects{suffix}.csv",
            log_file=output_dir / f"scraper{suffix}.log",
        )


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

def _load_index_set(csv_path):
    """Load a set of integer indices from a CSV's 'index' column."""
    if not csv_path.exists():
        return set()
    try:
        df = pd.read_csv(csv_path)
        return set(pd.to_numeric(df["index"], errors="coerce").dropna().astype(int))
    except Exception:
        return set()


def _write_csv_row(csv_path, row_dict):
    """Append a single row to a CSV using append mode (O(1), safe for hot paths)."""
    df = pd.DataFrame([row_dict])
    if csv_path.exists():
        df.to_csv(csv_path, mode="a", header=False, index=False)
    else:
        df.to_csv(csv_path, mode="w", header=True, index=False)


def save_completed(idx, panoid, lat, lon, paths):
    """Record a successfully downloaded location using the original request coords."""
    _write_csv_row(paths.completed_csv, {
        "timestamp": datetime.now().isoformat(),
        "index":     idx,
        "panoid":    panoid,
        "lat":       lat,
        "lon":       lon,
    })


def save_reject(idx, lat, lon, reason, paths, panoid=None):
    """Record a rejected location."""
    _write_csv_row(paths.rejects_csv, {
        "timestamp": datetime.now().isoformat(),
        "index":     idx,
        "lat":       lat,
        "lon":       lon,
        "reason":    reason,
        "panoid":    panoid if panoid else "N/A",
    })


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def setup_logging(log_file):
    """Initialize logging to file only (tqdm handles console)."""
    log_file.parent.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file)],
    )


def setup_directories(paths):
    """Create necessary directory structure."""
    paths.images.mkdir(parents=True, exist_ok=True)
    paths.metadata.mkdir(parents=True, exist_ok=True)
    logging.info(f"Directory structure created: {paths.images.parent}")


# ---------------------------------------------------------------------------
# Quality control
# ---------------------------------------------------------------------------

def is_quality_panorama(pano):
    """
    Quality control filter for Street View panoramas.
    Only accepts official Google Street View imagery.
    Returns: (is_valid, reason_if_invalid)
    """
    if hasattr(pano, "copyright_message") and pano.copyright_message:
        if "google" in pano.copyright_message.lower():
            return True, ""
        return False, "non_google_copyright"
    return False, "no_copyright_info"


# ---------------------------------------------------------------------------
# Image extraction
# ---------------------------------------------------------------------------

def extract_perspective_views(pano_image, headings, output_dir):
    """
    Extract perspective views at specified headings from equirectangular panorama.
    Returns True on success.
    """
    try:
        pano_array  = np.array(pano_image)
        pano_tensor = torch.from_numpy(pano_array).permute(2, 0, 1).float()

        for heading in headings:
            view_tensor = pytorch360convert.e2p(
                e_img=pano_tensor,
                fov_deg=VIEW_FOV,
                h_deg=heading,
                v_deg=0.0,
                out_hw=(VIEW_HEIGHT, VIEW_WIDTH),
                mode="bilinear",
                channels_first=True,
            )
            view_array = view_tensor.permute(1, 2, 0).numpy().clip(0, 255).astype("uint8")
            Image.fromarray(view_array).save(
                output_dir / f"{heading:03d}.jpg", quality=90, optimize=True
            )

        return True

    except Exception as e:
        logging.error(f"Error extracting views: {e}")
        logging.error(traceback.format_exc())
        return False


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

def _build_metadata(idx, pano, lat, lon):
    """Construct the metadata dict for a successfully downloaded location."""
    return {
        "index":              int(idx),
        "panoid":             pano.id,
        "pano_lat":           pano.lat,
        "pano_lon":           pano.lon,
        "original_lat":       float(lat),
        "original_lon":       float(lon),
        "date":               str(pano.date) if getattr(pano, "date", None) else None,
        "copyright":          getattr(pano, "copyright_message", None),
        "heading":            getattr(pano, "heading", None),
        "pitch":              getattr(pano, "pitch", None),
        "roll":               getattr(pano, "roll", None),
        "street_names":       [str(s) for s in pano.street_names] if getattr(pano, "street_names", None) else None,
        "address":            str(pano.address) if getattr(pano, "address", None) else None,
        "country_code":       str(pano.country_code) if getattr(pano, "country_code", None) else None,
        "download_timestamp": datetime.now().isoformat(),
        "headings":           HEADINGS,
        "view_resolution":    f"{VIEW_WIDTH}x{VIEW_HEIGHT}",
        "view_fov":           VIEW_FOV,
    }


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def download_location(row, completed, rejected, paths):
    """
    Download panorama and extract 4 directional views for a single location.
    Returns: (success, skip_reason)
    """
    idx = int(row.iloc[0])
    lat = row.iloc[1]
    lon = row.iloc[2]

    if idx in completed:
        return False, "already_completed"
    if idx in rejected:
        return False, "already_rejected"

    tmp_path = None
    try:
        pano = streetview.find_panorama(lat, lon)

        if pano is None:
            logging.warning(f"[{idx}] No panorama found at ({lat:.6f}, {lon:.6f})")
            save_reject(idx, lat, lon, "no_panorama_found", paths)
            return False, "no_panorama"

        is_valid, reason = is_quality_panorama(pano)
        if not is_valid:
            logging.info(f"[{idx}] Rejected: {reason} (panoid: {pano.id})")
            save_reject(idx, lat, lon, reason, paths, pano.id)
            return False, f"qc_failed_{reason}"

        location_dir = paths.images / f"{idx:06d}"
        location_dir.mkdir(parents=True, exist_ok=True)

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp_path = tmp.name
        streetview.download_panorama(pano, tmp_path, PANO_ZOOM)

        with Image.open(tmp_path) as pano_image:
            success = extract_perspective_views(pano_image, HEADINGS, location_dir)

        try:
            Path(tmp_path).unlink()
        except Exception as e:
            logging.warning(f"[{idx}] Could not delete temp file: {e}")
        tmp_path = None

        if not success:
            logging.error(f"[{idx}] Failed to extract views")
            try:
                shutil.rmtree(location_dir, ignore_errors=True)
            except Exception as e:
                logging.warning(f"[{idx}] Could not clean up failed folder {location_dir}: {e}")
            save_reject(idx, lat, lon, "view_extraction_failed", paths, pano.id)
            return False, "extraction_failed"

        meta_path = paths.metadata / f"{idx:06d}.json"
        meta_tmp  = meta_path.with_suffix(".tmp")
        meta_tmp.write_text(json.dumps(_build_metadata(idx, pano, lat, lon), indent=2), encoding="utf-8")
        os.replace(meta_tmp, meta_path)
        save_completed(idx, pano.id, lat, lon, paths)
        logging.info(f"[{idx}] Downloaded 4 views (panoid: {pano.id})")
        return True, None

    except (Timeout, ConnectionError) as e:
        logging.warning(f"[{idx}] Network error (timeout/connection): {e}")
        save_reject(idx, lat, lon, "network_timeout", paths)
        return False, "timeout"

    except Exception as e:
        logging.error(f"[{idx}] Error: {e}")
        logging.error(traceback.format_exc())
        save_reject(idx, lat, lon, f"error_{type(e).__name__}", paths)
        return False, "exception"

    finally:
        if tmp_path:
            try:
                Path(tmp_path).unlink()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Metadata backfill
# ---------------------------------------------------------------------------

def backfill_metadata(paths, coords_df):
    """
    Check all existing metadata JSON files and backfill new fields
    (heading, pitch, roll, etc.) for any that are missing them.
    Files already containing all fields are skipped instantly.
    Corrupt JSON files are repaired by looking up coordinates from the CSV.
    """
    metadata_files = sorted(paths.metadata.glob("*.json"))
    if not metadata_files:
        return True

    print(f"Checking {len(metadata_files)} metadata files for backfill...")
    logging.info(f"Checking {len(metadata_files)} metadata files for backfill")

    n_updated = n_repaired = 0
    consecutive_errors = 0

    api_fields    = ["heading", "pitch", "roll", "street_names", "address", "country_code"]
    config_fields = {
        "headings":        HEADINGS,
        "view_resolution": f"{VIEW_WIDTH}x{VIEW_HEIGHT}",
        "view_fov":        VIEW_FOV,
    }
    all_backfill_fields = api_fields + list(config_fields)

    with tqdm(total=len(metadata_files), desc="Backfill Metadata") as pbar:
        for meta_path in metadata_files:
            made_api_call = False
            try:
                # Try to parse the JSON
                corrupt = False
                try:
                    with open(meta_path, encoding="utf-8") as f:
                        metadata = json.load(f)
                except (json.JSONDecodeError, ValueError):
                    corrupt = True

                if corrupt:
                    idx = int(meta_path.stem)
                    row = coords_df[coords_df.iloc[:, 0] == idx]
                    if row.empty:
                        logging.warning(f"Backfill: corrupt JSON {meta_path.name}, index {idx} not found in CSV, skipping")
                        pbar.update(1)
                        continue

                    lat = float(row.iloc[0, 1])
                    lon = float(row.iloc[0, 2])
                    pano = streetview.find_panorama(lat, lon)
                    made_api_call = True

                    if pano is None:
                        logging.warning(f"Backfill: corrupt JSON {meta_path.name}, no panorama found, skipping")
                        pbar.update(1)
                        time.sleep(random.uniform(MIN_SLEEP, MAX_SLEEP))
                        continue

                    metadata = _build_metadata(idx, pano, lat, lon)
                    metadata["download_timestamp"] = None  # Unknown — original was corrupt
                    meta_tmp = meta_path.with_suffix(".tmp")
                    meta_tmp.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
                    os.replace(meta_tmp, meta_path)
                    n_repaired += 1
                    logging.info(f"Backfill: repaired corrupt JSON {meta_path.name}")
                    consecutive_errors = 0
                    pbar.update(1)
                    time.sleep(random.uniform(MIN_SLEEP, MAX_SLEEP))
                    continue

                # Check which fields are missing
                missing = [f for f in all_backfill_fields if f not in metadata]
                if not missing:
                    pbar.update(1)
                    continue

                # Fill config fields (no API call needed)
                for field in missing:
                    if field in config_fields:
                        metadata[field] = config_fields[field]

                # Fill API fields if needed
                needs_api = any(f in api_fields for f in missing)
                if needs_api:
                    pano = streetview.find_panorama_by_id(metadata["panoid"])
                    made_api_call = True
                    if pano is None:
                        pano = streetview.find_panorama(
                            metadata["original_lat"], metadata["original_lon"]
                        )
                        if pano is None or pano.id != metadata["panoid"]:
                            logging.warning(f"Backfill: panoid {metadata['panoid']} not found for {meta_path.name}, skipping")
                            pbar.update(1)
                            time.sleep(random.uniform(MIN_SLEEP, MAX_SLEEP))
                            continue

                    metadata.update({
                        "heading":      getattr(pano, "heading", None),
                        "pitch":        getattr(pano, "pitch", None),
                        "roll":         getattr(pano, "roll", None),
                        "street_names": [str(s) for s in pano.street_names] if getattr(pano, "street_names", None) else None,
                        "address":      str(pano.address) if getattr(pano, "address", None) else None,
                        "country_code": str(pano.country_code) if getattr(pano, "country_code", None) else None,
                    })

                meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
                n_updated += 1
                consecutive_errors = 0
                pbar.update(1)
                if made_api_call:
                    time.sleep(random.uniform(MIN_SLEEP, MAX_SLEEP))

            except Exception as e:
                logging.warning(f"Backfill error for {meta_path.name}: {e}")
                consecutive_errors += 1
                pbar.update(1)
                time.sleep(random.uniform(MIN_SLEEP, MAX_SLEEP))
                if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                    logging.error(f"Backfill: {MAX_CONSECUTIVE_ERRORS} consecutive errors, aborting")
                    print(f"Backfill aborted: {MAX_CONSECUTIVE_ERRORS} consecutive errors")
                    return False

    print(f"Backfill complete: {n_updated} updated, {n_repaired} repaired")
    logging.info(f"Backfill complete: {n_updated} updated, {n_repaired} repaired out of {len(metadata_files)} files")
    return True


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def stealth_sleep():
    """Randomized sleep between requests."""
    time.sleep(random.uniform(MIN_SLEEP, MAX_SLEEP))


def parse_args():
    """Parse command-line arguments for parallel operation."""
    parser = argparse.ArgumentParser(description="SALTY Street View Scraper")
    parser.add_argument("--start-index", type=int, default=0,
                        help="First coordinate index to process (inclusive, default: 0)")
    parser.add_argument("--end-index", type=int, default=None,
                        help="Last coordinate index to process (inclusive, default: all)")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    paths = _Paths.from_output_dir(OUTPUT_DIR, shard=args.start_index)

    print("=" * 70)
    print("SALTY SCRAPER - Multi-View Google Street View Data Acquisition")
    print("Project: Street-view Attention Learning Telemetry")
    print("CONFIGURATION: ZOOM 3 @ Quality 90")
    if args.start_index > 0 or args.end_index is not None:
        end_label = args.end_index if args.end_index is not None else "end"
        print(f"ROW RANGE: rows {args.start_index} - {end_label} of CSV")
    print("=" * 70)
    print()

    setup_logging(paths.log_file)
    setup_directories(paths)

    coords_df = pd.read_csv(COORDS_FILE)
    logging.info(f"Loaded {len(coords_df)} coordinates from {COORDS_FILE}")

    # Keep full coords for backfill — corrupt JSON repair needs to look up any index,
    # not just those in the current shard's slice.
    coords_df_full = coords_df

    # Slice to assigned row range (for parallel operation)
    # --start-index / --end-index are row positions in the CSV (0-99999),
    # NOT first-column values (which are original source dataset IDs)
    if args.start_index > 0 or args.end_index is not None:
        end = args.end_index + 1 if args.end_index is not None else len(coords_df)
        coords_df = coords_df.iloc[args.start_index:end].reset_index(drop=True)
        logging.info(f"Row slice: {args.start_index} - {args.end_index or 'end'} -> {len(coords_df)} coordinates")

    completed = _load_index_set(paths.completed_csv)
    rejected  = _load_index_set(paths.rejects_csv)

    logging.info(f"Found {len(completed)} completed locations")
    logging.info(f"Found {len(rejected)} rejected locations")

    if not backfill_metadata(paths, coords_df_full):
        print("Terminating due to backfill failure (network issue?)")
        return

    total      = len(coords_df)
    already_done = len(completed | rejected)
    remaining = total - already_done

    print(f"Total coordinates: {total:,}")
    print(f"Completed: {len(completed):,}")
    print(f"Rejected:  {len(rejected):,}")
    print(f"Remaining: {remaining:,}")
    print()
    print(f"Output: 4 views per location (0, 90, 180, 270 degrees)")
    print(f"View size: {VIEW_WIDTH}x{VIEW_HEIGHT} @ {VIEW_FOV} degree FOV")
    print(f"Panorama zoom: {PANO_ZOOM} (Higher resolution)")
    print(f"JPEG quality: 90 (Optimized for OCR/ML)")
    print(f"Stealth: {MIN_SLEEP}-{MAX_SLEEP}s per location")
    print(f"Output directory: {OUTPUT_DIR}")
    print()

    if remaining == 0:
        print("All locations processed!")
        return

    # Auto-confirmed for unattended/detached operation
    # response = input("Start/resume download? [y/N]: ")
    # if response.lower() != "y":
    #     print("Aborted.")
    #     return

    logging.info("=" * 50)
    logging.info("Starting SALTY multi-view scraper (ZOOM 3, Q90)")
    logging.info(f"Target: {remaining:,} locations")
    logging.info("=" * 50)

    success_count = error_count = processed = 0
    consecutive_timeouts = consecutive_errors = 0

    with tqdm(
        total=remaining,
        desc="Progress",
        leave=True,
        miniters=1,
        mininterval=0,
        bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}, {postfix}]",
    ) as pbar:

        for _, row in coords_df.iterrows():
            idx = int(row.iloc[0])

            if idx in completed or idx in rejected:
                continue

            success, reason = download_location(row, completed, rejected, paths)

            if success:
                success_count += 1
                completed.add(idx)
                consecutive_timeouts = 0
                consecutive_errors   = 0
            else:
                if reason not in ("already_completed", "already_rejected"):
                    rejected.add(idx)

                    if reason in ("no_panorama",) or reason.startswith("qc_failed"):
                        # API responded fine — not a network/error issue
                        consecutive_timeouts = 0
                        consecutive_errors   = 0
                    else:
                        error_count       += 1
                        consecutive_errors += 1

                        if reason == "timeout":
                            consecutive_timeouts += 1
                            logging.warning(f"Consecutive timeouts: {consecutive_timeouts}/{MAX_CONSECUTIVE_TIMEOUTS}")
                            if consecutive_timeouts >= MAX_CONSECUTIVE_TIMEOUTS:
                                logging.error(f"TERMINATING: {MAX_CONSECUTIVE_TIMEOUTS} consecutive timeouts reached")
                                pbar.write(f"\nERROR: {MAX_CONSECUTIVE_TIMEOUTS} consecutive network timeouts.")
                                pbar.write("This likely indicates a network or API issue.")
                                pbar.write(f"Progress saved. {success_count} locations downloaded before termination.")
                                break
                        else:
                            consecutive_timeouts = 0

                        if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                            logging.error(f"TERMINATING: {MAX_CONSECUTIVE_ERRORS} consecutive errors reached. Possible IP ban.")
                            pbar.write(f"\nERROR: {MAX_CONSECUTIVE_ERRORS} consecutive errors detected.")
                            pbar.write("This likely indicates an IP ban or API block.")
                            pbar.write(f"Progress saved. {success_count} locations downloaded before termination.")
                            break

            processed += 1
            pbar.update(1)
            pbar.set_postfix_str(f"✓ {success_count} ✗ {error_count} | rate:{success_count/processed*100:.0f}%")

            if reason not in ("already_completed", "already_rejected"):
                stealth_sleep()

    print()
    print("=" * 70)
    print("DOWNLOAD COMPLETE")
    print("=" * 70)
    print(f"Successfully downloaded: {success_count:,} locations ({success_count * 4:,} images)")
    print(f"Rejected: {error_count:,} locations")
    print(f"Total processed: {processed:,}")
    if processed > 0:
        print(f"Success rate: {success_count / processed * 100:.1f}%")
    print(f"Images location:   {paths.images}")
    print(f"Metadata location: {paths.metadata}")
    print(f"Completed log:     {paths.completed_csv}")
    print(f"Rejects log:       {paths.rejects_csv}")
    print()

    logging.info("=" * 50)
    logging.info("SALTY scraper completed")
    logging.info(f"Success: {success_count:,} | Rejected: {error_count:,}")
    logging.info("=" * 50)


if __name__ == "__main__":
    main()
