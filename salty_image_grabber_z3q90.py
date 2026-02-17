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

import streetlevel.streetview as streetview
import pandas as pd
import numpy as np
import time
import random
import argparse
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import json
import logging
import traceback
from PIL import Image
import pytorch360convert
import torch
import tempfile
from requests.exceptions import Timeout, ConnectionError

# Configuration
COORDS_FILE = "100k-205k_data.csv"
OUTPUT_DIR = Path("salty_data")
IMAGES_DIR = OUTPUT_DIR / "images"
METADATA_DIR = OUTPUT_DIR / "metadata"
COMPLETED_FILE = OUTPUT_DIR / "completed.csv"
REJECTS_FILE = OUTPUT_DIR / "rejects.csv"
LOG_FILE = OUTPUT_DIR / "scraper.log"

# Download settings
PANO_ZOOM = 3  # Zoom level for downloading equirectangular panorama (HIGHER RESOLUTION)
VIEW_HEIGHT = 1024  # Height of extracted perspective views
VIEW_WIDTH = 1024   # Width of extracted perspective views
VIEW_FOV = 90.0     # Field of view for perspective views (degrees)
HEADINGS = [0, 90, 180, 270]  # Cardinal directions to extract

# Stealth settings
MIN_SLEEP = 0.25   # Minimum seconds between requests
MAX_SLEEP = 1.0  # Maximum seconds between requests

# Error handling
MAX_CONSECUTIVE_TIMEOUTS = 5  # Terminate after this many consecutive timeouts
MAX_CONSECUTIVE_ERRORS = 10   # Terminate after this many consecutive errors of ANY type (likely IP ban)

# Setup logging
def setup_logging():
    """Initialize logging to file only (tqdm handles console)."""
    OUTPUT_DIR.mkdir(exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(LOG_FILE)
        ]
    )

def setup_directories():
    """Create necessary directory structure."""
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    logging.info(f"Directory structure created: {OUTPUT_DIR}")

def load_coordinates():
    """Load the 100k California coordinate dataset."""
    df = pd.read_csv(COORDS_FILE)
    logging.info(f"Loaded {len(df)} coordinates from {COORDS_FILE}")
    logging.info(f"Columns: {list(df.columns)}")
    return df

def load_completed():
    """Load set of completed location indices."""
    if COMPLETED_FILE.exists():
        df = pd.read_csv(COMPLETED_FILE)
        completed = set(df['index'].values)
        logging.info(f"Found {len(completed)} completed locations")
        return completed
    else:
        logging.info("No completed locations found (fresh start)")
        return set()

def load_rejects():
    """Load set of rejected location indices."""
    if REJECTS_FILE.exists():
        df = pd.read_csv(REJECTS_FILE)
        rejects = set(df['index'].values)
        logging.info(f"Found {len(rejects)} rejected locations")
        return rejects
    else:
        logging.info("No rejected locations found")
        return set()

def save_completed(index, panoid, lat, lon):
    """Append completed location to CSV."""
    entry = {
        'timestamp': datetime.now().isoformat(),
        'index': index,
        'panoid': panoid,
        'lat': lat,
        'lon': lon
    }

    df = pd.DataFrame([entry])
    if COMPLETED_FILE.exists():
        df.to_csv(COMPLETED_FILE, mode='a', header=False, index=False)
    else:
        df.to_csv(COMPLETED_FILE, mode='w', header=True, index=False)

def save_reject(index, lat, lon, reason, panoid=None):
    """Append rejected location to CSV."""
    entry = {
        'timestamp': datetime.now().isoformat(),
        'index': index,
        'lat': lat,
        'lon': lon,
        'reason': reason,
        'panoid': panoid if panoid else 'N/A'
    }

    df = pd.DataFrame([entry])
    if REJECTS_FILE.exists():
        df.to_csv(REJECTS_FILE, mode='a', header=False, index=False)
    else:
        df.to_csv(REJECTS_FILE, mode='w', header=True, index=False)

def is_quality_panorama(pano):
    """
    Quality control filter for Street View panoramas.

    Only accepts official Google Street View imagery.

    Returns:
        tuple: (is_valid, reason_if_invalid)
    """
    # Only accept images with Google copyright
    if hasattr(pano, 'copyright_message') and pano.copyright_message:
        if 'google' in pano.copyright_message.lower():
            return True, ""
        else:
            return False, "non_google_copyright"

    # If no copyright message, reject to be safe
    return False, "no_copyright_info"

def extract_perspective_views(pano_image, headings, output_dir):
    """
    Extract perspective views at specified headings from equirectangular panorama.

    Args:
        pano_image: PIL Image of equirectangular panorama
        headings: List of heading angles in degrees
        output_dir: Path to save extracted views

    Returns:
        bool: Success status
    """
    try:
        # Convert PIL image to numpy array then to torch tensor
        pano_array = np.array(pano_image)

        # Convert to torch tensor: (H, W, C) -> (C, H, W)
        pano_tensor = torch.from_numpy(pano_array).permute(2, 0, 1).float()

        # Extract view at each heading
        for heading in headings:
            # Extract perspective view using pytorch360convert
            view_tensor = pytorch360convert.e2p(
                e_img=pano_tensor,
                fov_deg=VIEW_FOV,
                h_deg=heading,
                v_deg=0.0,
                out_hw=(VIEW_HEIGHT, VIEW_WIDTH),
                mode='bilinear',
                channels_first=True
            )

            # Convert back to numpy: (C, H, W) -> (H, W, C)
            view_array = view_tensor.permute(1, 2, 0).numpy().astype('uint8')

            # Convert to PIL Image and save
            view_image = Image.fromarray(view_array)
            output_path = output_dir / f"{heading:03d}.jpg"
            view_image.save(output_path, quality=90, optimize=True)

        return True

    except Exception as e:
        logging.error(f"Error extracting views: {str(e)}")
        logging.error(traceback.format_exc())
        return False

def download_location(row, completed_indices, rejected_indices):
    """
    Download panorama and extract 4 directional views for a single location.

    Args:
        row: DataFrame row containing coordinate data
        completed_indices: Set of already completed indices
        rejected_indices: Set of already rejected indices

    Returns:
        tuple: (success, skip_reason)
            success: True if downloaded successfully
            skip_reason: Reason for skip/failure if applicable
    """
    idx = int(row.iloc[0])  # First column is index
    lat = row.iloc[1]  # Second column is latitude
    lon = row.iloc[2]  # Third column is longitude

    # Skip if already completed
    if idx in completed_indices:
        return False, "already_completed"

    # Skip if already rejected
    if idx in rejected_indices:
        return False, "already_rejected"

    tmp_path = None
    try:
        # Find nearest panorama
        pano = streetview.find_panorama(lat, lon)

        if pano is None:
            logging.warning(f"[{idx}] No panorama found at ({lat:.6f}, {lon:.6f})")
            save_reject(idx, lat, lon, "no_panorama_found")
            return False, "no_panorama"

        # Quality control check
        is_valid, reason = is_quality_panorama(pano)
        if not is_valid:
            logging.info(f"[{idx}] Rejected: {reason} (panoid: {pano.id})")
            save_reject(idx, lat, lon, reason, pano.id)
            return False, f"qc_failed_{reason}"

        # Create location directory
        location_dir = IMAGES_DIR / f"{idx:06d}"
        location_dir.mkdir(parents=True, exist_ok=True)

        # Download equirectangular panorama to temporary file
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            tmp_path = tmp.name
            streetview.download_panorama(pano, tmp_path, PANO_ZOOM)

        # Load into PIL Image (outside with block)
        pano_image = Image.open(tmp_path)

        # Extract 4 directional views
        success = extract_perspective_views(pano_image, HEADINGS, location_dir)

        # Close image to release file handle before deleting
        pano_image.close()

        # Clean up temp panorama file (we only keep the 4 extracted views)
        try:
            Path(tmp_path).unlink()
        except Exception as e:
            logging.warning(f"[{idx}] Could not delete temp file: {e}")
        tmp_path = None

        if not success:
            logging.error(f"[{idx}] Failed to extract views")
            save_reject(idx, lat, lon, "view_extraction_failed", pano.id)
            return False, "extraction_failed"

        # Save metadata
        metadata = {
            'index': int(idx),
            'panoid': pano.id,
            'pano_lat': pano.lat,
            'pano_lon': pano.lon,
            'original_lat': float(lat),
            'original_lon': float(lon),
            'date': str(pano.date) if hasattr(pano, 'date') and pano.date else None,
            'copyright': getattr(pano, 'copyright_message', None),
            'heading': getattr(pano, 'heading', None),
            'pitch': getattr(pano, 'pitch', None),
            'roll': getattr(pano, 'roll', None),
            'street_names': [str(s) for s in pano.street_names] if getattr(pano, 'street_names', None) else None,
            'address': str(pano.address) if getattr(pano, 'address', None) else None,
            'country_code': str(pano.country_code) if getattr(pano, 'country_code', None) else None,
            'download_timestamp': datetime.now().isoformat(),
            'headings': HEADINGS,
            'view_resolution': f"{VIEW_WIDTH}x{VIEW_HEIGHT}",
            'view_fov': VIEW_FOV
        }

        metadata_path = METADATA_DIR / f"{idx:06d}.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        # Mark as completed
        save_completed(idx, pano.id, pano.lat, pano.lon)
        logging.info(f"[{idx}] Downloaded 4 views (panoid: {pano.id})")

        return True, None

    except (Timeout, ConnectionError) as e:
        logging.warning(f"[{idx}] Network error (timeout/connection): {str(e)}")
        save_reject(idx, lat, lon, "network_timeout")
        return False, "timeout"

    except Exception as e:
        logging.error(f"[{idx}] Error: {str(e)}")
        logging.error(traceback.format_exc())
        save_reject(idx, lat, lon, f"error_{type(e).__name__}")
        return False, "exception"

    finally:
        # Clean up temp file if it wasn't already deleted
        if tmp_path:
            try:
                Path(tmp_path).unlink()
            except Exception:
                pass

def backfill_metadata():
    """
    Check all existing metadata JSON files and backfill new fields
    (heading, pitch, roll, etc.) for any that are missing them.
    Files already containing the new fields are skipped instantly.
    Corrupt JSON files are repaired by looking up coordinates from the CSV.
    """
    metadata_files = sorted(METADATA_DIR.glob("*.json"))
    if not metadata_files:
        return True

    # Load coordinates CSV for repairing corrupt metadata
    coords_df = pd.read_csv(COORDS_FILE)

    print(f"Checking {len(metadata_files)} metadata files for backfill...")
    logging.info(f"Checking {len(metadata_files)} metadata files for backfill")
    updated = 0
    repaired = 0
    consecutive_errors = 0

    with tqdm(total=len(metadata_files), desc="Backfill Metadata") as pbar:
        for meta_path in metadata_files:
            try:
                # Try to parse the JSON
                corrupt = False
                try:
                    with open(meta_path, 'r') as f:
                        metadata = json.load(f)
                except (json.JSONDecodeError, ValueError):
                    corrupt = True

                if corrupt:
                    # Repair: extract index from filename, look up coords in CSV
                    idx = int(meta_path.stem)  # e.g. "000151" -> 151
                    row = coords_df[coords_df.iloc[:, 0] == idx]
                    if row.empty:
                        logging.warning(f"Backfill: corrupt JSON {meta_path.name}, index {idx} not found in CSV, skipping")
                        pbar.update(1)
                        continue

                    lat = float(row.iloc[0, 1])
                    lon = float(row.iloc[0, 2])

                    pano = streetview.find_panorama(lat, lon)
                    if pano is None:
                        logging.warning(f"Backfill: corrupt JSON {meta_path.name}, no panorama found, skipping")
                        pbar.update(1)
                        time.sleep(random.uniform(MIN_SLEEP, MAX_SLEEP))
                        continue

                    # Rebuild the full metadata from scratch
                    metadata = {
                        'index': idx,
                        'panoid': pano.id,
                        'pano_lat': pano.lat,
                        'pano_lon': pano.lon,
                        'original_lat': lat,
                        'original_lon': lon,
                        'date': str(pano.date) if hasattr(pano, 'date') and pano.date else None,
                        'copyright': getattr(pano, 'copyright_message', None),
                        'heading': getattr(pano, 'heading', None),
                        'pitch': getattr(pano, 'pitch', None),
                        'roll': getattr(pano, 'roll', None),
                        'street_names': [str(s) for s in pano.street_names] if getattr(pano, 'street_names', None) else None,
                        'address': str(pano.address) if getattr(pano, 'address', None) else None,
                        'country_code': str(pano.country_code) if getattr(pano, 'country_code', None) else None,
                        'download_timestamp': None,  # Unknown — original was corrupt
                        'headings': HEADINGS,
                        'view_resolution': f"{VIEW_WIDTH}x{VIEW_HEIGHT}",
                        'view_fov': VIEW_FOV
                    }

                    with open(meta_path, 'w') as f:
                        json.dump(metadata, f, indent=2)
                    repaired += 1
                    logging.info(f"Backfill: repaired corrupt JSON {meta_path.name}")
                    consecutive_errors = 0
                    pbar.update(1)
                    time.sleep(random.uniform(MIN_SLEEP, MAX_SLEEP))
                    continue

                # Check which fields are missing
                api_fields = ['heading', 'pitch', 'roll', 'street_names', 'address', 'country_code']
                config_fields = {'headings': HEADINGS, 'view_resolution': f"{VIEW_WIDTH}x{VIEW_HEIGHT}", 'view_fov': VIEW_FOV}
                fixable_fields = api_fields + list(config_fields.keys())

                missing = [f for f in fixable_fields if f not in metadata]
                if not missing:
                    pbar.update(1)
                    continue

                # Fill in config fields (no API call needed)
                needs_api = False
                for field in missing:
                    if field in config_fields:
                        metadata[field] = config_fields[field]
                    elif field in api_fields:
                        needs_api = True

                # Fill in API fields if needed
                if needs_api:
                    pano = streetview.find_panorama_by_id(metadata['panoid'])
                    if pano is None:
                        pano = streetview.find_panorama(
                            metadata['original_lat'],
                            metadata['original_lon']
                        )
                        if pano is None or pano.id != metadata['panoid']:
                            logging.warning(f"Backfill: panoid {metadata['panoid']} not found for {meta_path.name}, skipping")
                            pbar.update(1)
                            time.sleep(random.uniform(MIN_SLEEP, MAX_SLEEP))
                            continue

                    metadata['heading'] = getattr(pano, 'heading', None)
                    metadata['pitch'] = getattr(pano, 'pitch', None)
                    metadata['roll'] = getattr(pano, 'roll', None)
                    metadata['street_names'] = [str(s) for s in pano.street_names] if getattr(pano, 'street_names', None) else None
                    metadata['address'] = str(pano.address) if getattr(pano, 'address', None) else None
                    metadata['country_code'] = str(pano.country_code) if getattr(pano, 'country_code', None) else None

                with open(meta_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
                updated += 1

                consecutive_errors = 0
                pbar.update(1)
                if needs_api:
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

    print(f"Backfill complete: {updated} updated, {repaired} repaired")
    logging.info(f"Backfill complete: {updated} updated, {repaired} repaired out of {len(metadata_files)} files")
    return True

def stealth_sleep():
    """Randomized sleep between requests."""
    sleep_time = random.uniform(MIN_SLEEP, MAX_SLEEP)
    time.sleep(sleep_time)


def parse_args():
    """Parse command-line arguments for parallel operation."""
    parser = argparse.ArgumentParser(description='SALTY Street View Scraper')
    parser.add_argument('--start-index', type=int, default=0,
                        help='First coordinate index to process (inclusive, default: 0)')
    parser.add_argument('--end-index', type=int, default=None,
                        help='Last coordinate index to process (inclusive, default: all)')
    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_args()

    # Instance-specific file paths when running in parallel (start-index > 0)
    if args.start_index > 0:
        global COMPLETED_FILE, REJECTS_FILE, LOG_FILE
        COMPLETED_FILE = OUTPUT_DIR / f"completed_{args.start_index}.csv"
        REJECTS_FILE = OUTPUT_DIR / f"rejects_{args.start_index}.csv"
        LOG_FILE = OUTPUT_DIR / f"scraper_{args.start_index}.log"

    print("=" * 70)
    print("SALTY SCRAPER - Multi-View Google Street View Data Acquisition")
    print("Project: Street-view Attention Learning Telemetry")
    print("CONFIGURATION: ZOOM 3 @ Quality 90")
    if args.start_index > 0 or args.end_index is not None:
        end_label = args.end_index if args.end_index is not None else "end"
        print(f"ROW RANGE: rows {args.start_index} - {end_label} of CSV")
    print("=" * 70)
    print()

    # Setup
    setup_logging()
    setup_directories()

    # Load data
    coords_df = load_coordinates()

    # Slice to assigned row range (for parallel operation)
    # --start-index / --end-index are row positions in the CSV (0-99999),
    # NOT first-column values (which are original source dataset IDs)
    if args.start_index > 0 or args.end_index is not None:
        end = args.end_index + 1 if args.end_index is not None else len(coords_df)
        coords_df = coords_df.iloc[args.start_index:end].reset_index(drop=True)
        logging.info(f"Row slice: {args.start_index} - {args.end_index or 'end'} → {len(coords_df)} coordinates")

    completed = load_completed()
    rejected = load_rejects()

    # Backfill old metadata files with new fields (skips files already updated)
    if not backfill_metadata():
        print("Terminating due to backfill failure (network issue?)")
        return

    # Statistics
    total = len(coords_df)
    already_done = len(completed) + len(rejected)
    remaining = total - already_done

    print(f"Total coordinates: {total:,}")
    print(f"Completed: {len(completed):,}")
    print(f"Rejected: {len(rejected):,}")
    print(f"Remaining: {remaining:,}")
    print(f"")
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

    # Confirmation (auto-confirmed for unattended operation)
    response = 'y'  # Auto-confirm for detached mode
    # response = input("Start/resume download? [y/N]: ")  # Commented out for unattended runs
    if response.lower() != 'y':
        print("Aborted.")
        return

    print()
    logging.info("="*50)
    logging.info("Starting SALTY multi-view scraper (ZOOM 3, Q90)")
    logging.info(f"Target: {remaining:,} locations")
    logging.info("="*50)

    # Download loop
    success_count = 0
    error_count = 0
    processed = 0
    consecutive_timeouts = 0
    consecutive_errors = 0

    with tqdm(
        total=remaining,
        desc="Progress",
        leave=True,
        miniters=1,
        mininterval=0,
        bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}, {postfix}]'
    ) as pbar:

        for _, row in coords_df.iterrows():
            idx = int(row.iloc[0])

            # Skip if already processed
            if idx in completed or idx in rejected:
                continue

            # Download location with 4 views
            success, reason = download_location(row, completed, rejected)

            if success:
                success_count += 1
                completed.add(idx)
                consecutive_timeouts = 0  # Reset on success
                consecutive_errors = 0    # Reset on success
            else:
                if reason not in ["already_completed", "already_rejected"]:
                    rejected.add(idx)

                    # Non-error rejections (API worked, just no usable panorama)
                    if reason in ["no_panorama"] or reason.startswith("qc_failed"):
                        consecutive_timeouts = 0  # API responded, network is fine
                        consecutive_errors = 0
                    else:
                        error_count += 1
                        consecutive_errors += 1

                        # Track consecutive timeouts
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
                            consecutive_timeouts = 0  # Reset on non-timeout error

                        # Track consecutive errors of ANY type (IP ban detection)
                        if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                            logging.error(f"TERMINATING: {MAX_CONSECUTIVE_ERRORS} consecutive errors reached. Possible IP ban.")
                            pbar.write(f"\nERROR: {MAX_CONSECUTIVE_ERRORS} consecutive errors detected.")
                            pbar.write("This likely indicates an IP ban or API block.")
                            pbar.write(f"Progress saved. {success_count} locations downloaded before termination.")
                            break

            processed += 1

            pbar.update(1)
            pbar.set_postfix_str(f"✓ {success_count} ✗ {error_count} | rate:{success_count/processed*100:.0f}%")

            # Stealth sleep (only if we actually tried to download)
            if reason not in ["already_completed", "already_rejected"]:
                stealth_sleep()

    # Final summary
    print()
    print("="*70)
    print("DOWNLOAD COMPLETE")
    print("="*70)
    print(f"Successfully downloaded: {success_count:,} locations ({success_count * 4:,} images)")
    print(f"Rejected: {error_count:,} locations")
    print(f"Total processed: {processed:,}")
    if processed > 0:
        print(f"Success rate: {success_count/processed*100:.1f}%")
    print(f"Images location: {IMAGES_DIR}")
    print(f"Metadata location: {METADATA_DIR}")
    print(f"Completed log: {COMPLETED_FILE}")
    print(f"Rejects log: {REJECTS_FILE}")
    print()

    logging.info("="*50)
    logging.info("SALTY scraper completed")
    logging.info(f"Success: {success_count:,} | Rejected: {error_count:,}")
    logging.info("="*50)

if __name__ == "__main__":
    main()
