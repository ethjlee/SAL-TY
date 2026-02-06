"""
SALTY Scraper - Google Street View Multi-View Data Acquisition
Project SALTY: Street-view Attention Learning Telemetry

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
- Randomized sleep between requests (5-10s)
- Sub-batch pause after 100 images (60s)
- Full batch checkpoint after 1000 images
"""

import streetlevel.streetview as streetview
import pandas as pd
import numpy as np
import time
import random
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import json
import logging
from PIL import Image
import pytorch360convert
import torch
import tempfile
from requests.exceptions import Timeout, ConnectionError

# Configuration
COORDS_FILE = "all_data.csv"
OUTPUT_DIR = Path("salty_data")
IMAGES_DIR = OUTPUT_DIR / "images"
METADATA_DIR = OUTPUT_DIR / "metadata"
COMPLETED_FILE = OUTPUT_DIR / "completed.csv"
REJECTS_FILE = OUTPUT_DIR / "rejects.csv"
LOG_FILE = OUTPUT_DIR / "scraper.log"

# Download settings
PANO_ZOOM = 2  # Zoom level for downloading equirectangular panorama
VIEW_HEIGHT = 1024  # Height of extracted perspective views
VIEW_WIDTH = 1024   # Width of extracted perspective views
VIEW_FOV = 90.0     # Field of view for perspective views (degrees)
HEADINGS = [0, 90, 180, 270]  # Cardinal directions to extract

# Stealth settings
MIN_SLEEP = 1.5   # Minimum seconds between requests
MAX_SLEEP = 3.5  # Maximum seconds between requests
SUB_BATCH = 250  # Checkpoint interval
SUB_BATCH_SLEEP = 15  # Pause after sub-batch (seconds)
FULL_BATCH = 2000  # Major checkpoint interval
FULL_BATCH_SLEEP = 60  # Pause after full batch (5 minutes)

# Error handling
MAX_CONSECUTIVE_TIMEOUTS = 5  # Terminate after this many consecutive timeouts

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

def extract_perspective_views(pano_image, headings, output_dir, index):
    """
    Extract perspective views at specified headings from equirectangular panorama.

    Args:
        pano_image: PIL Image of equirectangular panorama
        headings: List of heading angles in degrees
        output_dir: Path to save extracted views
        index: Location index for filenames

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
            view_image.save(output_path, quality=95)

        return True

    except Exception as e:
        logging.error(f"Error extracting views: {str(e)}")
        import traceback
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
        success = extract_perspective_views(pano_image, HEADINGS, location_dir, idx)

        # Close image to release file handle before deleting
        pano_image.close()

        # Clean up temp panorama file (we only keep the 4 extracted views)
        try:
            Path(tmp_path).unlink()
        except Exception as e:
            logging.warning(f"[{idx}] Could not delete temp file: {e}")

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
        import traceback
        logging.error(traceback.format_exc())
        save_reject(idx, lat, lon, f"error_{type(e).__name__}")
        return False, "exception"

def stealth_sleep(count, pbar=None):
    """
    Implement stealth protocol with randomized delays.

    Args:
        count: Number of locations processed so far
        pbar: Optional tqdm progress bar for clean output
    """
    # Base random sleep between requests
    sleep_time = random.uniform(MIN_SLEEP, MAX_SLEEP)
    time.sleep(sleep_time)

    # Sub-batch checkpoint with randomization
    if count > 0 and count % SUB_BATCH == 0:
        sub_batch_sleep = random.uniform(SUB_BATCH_SLEEP - 10, SUB_BATCH_SLEEP + 10)
        msg = f"Sub-batch checkpoint at {count} locations. Sleeping {sub_batch_sleep:.1f}s..."
        logging.info(msg)
        if pbar:
            pbar.write(f"\n[{count}] Sub-batch checkpoint - pausing {sub_batch_sleep:.0f}s...")
        time.sleep(sub_batch_sleep)

    # Full batch checkpoint with randomization
    if count > 0 and count % FULL_BATCH == 0:
        full_batch_sleep = random.uniform(FULL_BATCH_SLEEP - 30, FULL_BATCH_SLEEP + 30)
        msg = f"FULL BATCH CHECKPOINT at {count} locations. Sleeping {full_batch_sleep:.1f}s..."
        logging.info(msg)
        if pbar:
            pbar.write(f"\n[{count}] FULL BATCH CHECKPOINT - pausing {full_batch_sleep:.0f}s...")
        time.sleep(full_batch_sleep)

def main():
    """Main execution function."""
    print("=" * 70)
    print("SALTY SCRAPER - Multi-View Google Street View Data Acquisition")
    print("Project: Street-view Attention Learning Telemetry")
    print("=" * 70)
    print()

    # Setup
    setup_logging()
    setup_directories()

    # Load data
    coords_df = load_coordinates()
    completed = load_completed()
    rejected = load_rejects()

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
    print(f"Stealth: {MIN_SLEEP}-{MAX_SLEEP}s per location")
    print(f"Output directory: {OUTPUT_DIR}")
    print()

    if remaining == 0:
        print("All locations processed!")
        return

    # Confirmation
    response = input("Start/resume download? [y/N]: ")
    if response.lower() != 'y':
        print("Aborted.")
        return

    print()
    logging.info("="*50)
    logging.info("Starting SALTY multi-view scraper")
    logging.info(f"Target: {remaining:,} locations")
    logging.info("="*50)

    # Download loop
    success_count = 0
    error_count = 0
    processed = 0
    consecutive_timeouts = 0

    # Create nested progress bars - overall, full batch, sub-batch
    with tqdm(
        total=remaining,
        desc="Overall Progress",
        position=0,
        leave=True,
        bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
    ) as pbar_overall:
        with tqdm(
            total=FULL_BATCH,
            desc="Full Batch (1000)",
            position=1,
            leave=False,
            bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}'
        ) as pbar_full:
            with tqdm(
                total=SUB_BATCH,
                desc="Sub-Batch (100)",
                position=2,
                leave=False,
                bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{postfix}]'
            ) as pbar_sub:

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
                    else:
                        if reason not in ["already_completed", "already_rejected"]:
                            error_count += 1
                            rejected.add(idx)

                            # Track consecutive timeouts
                            if reason == "timeout":
                                consecutive_timeouts += 1
                                logging.warning(f"Consecutive timeouts: {consecutive_timeouts}/{MAX_CONSECUTIVE_TIMEOUTS}")

                                if consecutive_timeouts >= MAX_CONSECUTIVE_TIMEOUTS:
                                    logging.error(f"TERMINATING: {MAX_CONSECUTIVE_TIMEOUTS} consecutive timeouts reached")
                                    pbar_overall.write(f"\nERROR: {MAX_CONSECUTIVE_TIMEOUTS} consecutive network timeouts.")
                                    pbar_overall.write("This likely indicates a network or API issue.")
                                    pbar_overall.write(f"Progress saved. {success_count} locations downloaded before termination.")
                                    break
                            else:
                                consecutive_timeouts = 0  # Reset on non-timeout error

                    processed += 1

                    # Update all bars
                    pbar_overall.update(1)
                    pbar_full.update(1)
                    pbar_sub.update(1)

                    # Update sub-batch postfix
                    pbar_sub.set_postfix_str(f"✓ {success_count} ✗ {error_count} | rate:{success_count/processed*100:.0f}%")

                    # Reset sub-batch bar every 100
                    if success_count > 0 and success_count % SUB_BATCH == 0:
                        pbar_sub.n = 0
                        pbar_sub.refresh()

                    # Reset full batch bar every 1000
                    if success_count > 0 and success_count % FULL_BATCH == 0:
                        pbar_full.n = 0
                        pbar_full.refresh()

                    # Stealth sleep (only if we actually tried to download)
                    if reason not in ["already_completed", "already_rejected"]:
                        stealth_sleep(success_count, pbar_overall)

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
