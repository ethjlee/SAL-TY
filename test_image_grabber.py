"""
SALTY Scraper - TEST VERSION (10 locations)
Quick test of multi-view Street View download functionality
Tests extraction of 4 directional views per location
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

# Configuration - TEST MODE
COORDS_FILE = "all_data.csv"
OUTPUT_DIR = Path("test_output")
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

# TEST settings - lighter delays
MIN_SLEEP = 2   # Lighter for testing
MAX_SLEEP = 4   # Lighter for testing
TEST_COUNT = 10  # Only download 10 locations

# Error handling
MAX_CONSECUTIVE_TIMEOUTS = 3  # Terminate after this many consecutive timeouts (lower for testing)

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
    # TEST: Only take first 20 rows (to have buffer for rejections)
    df = df.head(20)
    logging.info(f"TEST MODE: Using first {len(df)} coordinates")
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

def main():
    """Main execution function."""
    print("=" * 70)
    print("SALTY SCRAPER - TEST MODE (10 locations)")
    print("Testing multi-view Street View download with quality control")
    print("=" * 70)
    print()

    # Setup
    setup_logging()
    setup_directories()

    # Load data
    coords_df = load_coordinates()
    completed = load_completed()
    rejected = load_rejects()

    print(f"Test mode: Downloading up to {TEST_COUNT} locations")
    print(f"Already completed: {len(completed)}")
    print(f"Already rejected: {len(rejected)}")
    print(f"Output: 4 views per location (0, 90, 180, 270 degrees)")
    print(f"View size: {VIEW_WIDTH}x{VIEW_HEIGHT} @ {VIEW_FOV} degree FOV")
    print(f"Sleep: {MIN_SLEEP}-{MAX_SLEEP}s per request (light for testing)")
    print(f"Output directory: {OUTPUT_DIR}")
    print()

    # Start download
    print("Starting test download...\n")
    logging.info("="*50)
    logging.info("Starting TEST scraper")
    logging.info(f"Target: {TEST_COUNT} locations")
    logging.info("="*50)

    # Download loop
    success_count = 0
    error_count = 0
    attempts = 0
    consecutive_timeouts = 0

    # Create simple progress bar for test mode
    with tqdm(
        total=TEST_COUNT,
        desc="Test Progress",
        position=0,
        leave=True,
        bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {postfix}]'
    ) as pbar:
        for _, row in coords_df.iterrows():
            idx = int(row.iloc[0])

            # Skip if already processed
            if idx in completed or idx in rejected:
                continue

            # Stop after TEST_COUNT successful downloads
            if success_count >= TEST_COUNT:
                break

            # Download location with 4 views
            attempts += 1
            success, reason = download_location(row, completed, rejected)

            if success:
                success_count += 1
                completed.add(idx)
                consecutive_timeouts = 0  # Reset on success
                pbar.update(1)
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
                            pbar.write(f"\nERROR: {MAX_CONSECUTIVE_TIMEOUTS} consecutive network timeouts.")
                            pbar.write("This likely indicates a network or API issue.")
                            pbar.write(f"Progress saved. {success_count} locations downloaded before termination.")
                            break
                    else:
                        consecutive_timeouts = 0  # Reset on non-timeout error

            pbar.set_postfix_str(f"✓ {success_count} ✗ {error_count} | rate:{success_count/attempts*100:.0f}%")

            # Light sleep for testing
            if success_count < TEST_COUNT and reason not in ["already_completed", "already_rejected"]:
                sleep_time = random.uniform(MIN_SLEEP, MAX_SLEEP)
                time.sleep(sleep_time)

    # Final summary
    print()
    print("="*70)
    print("TEST COMPLETE")
    print("="*70)
    print(f"Successfully downloaded: {success_count} locations ({success_count * 4} images)")
    print(f"Errors/Rejected: {error_count}")
    print(f"Total attempts: {attempts}")
    if attempts > 0:
        print(f"Success rate: {success_count/attempts*100:.1f}%")
    else:
        print("Success rate: N/A (all locations already processed)")
    print(f"Output location: {IMAGES_DIR}")
    print(f"Metadata location: {METADATA_DIR}")
    print(f"Completed log: {COMPLETED_FILE}")
    print(f"Rejects log: {REJECTS_FILE}")
    print()

    # Show downloaded folders and files
    if success_count > 0:
        print("Downloaded locations:")
        for location_dir in sorted(IMAGES_DIR.glob("*")):
            if location_dir.is_dir():
                views = list(location_dir.glob("*.jpg"))
                total_size_kb = sum(f.stat().st_size for f in views) / 1024
                print(f"  {location_dir.name}/ - {len(views)} views ({total_size_kb:.1f} KB)")

    logging.info("="*50)
    logging.info("TEST scraper completed")
    logging.info(f"Success: {success_count} | Errors: {error_count}")
    logging.info("="*50)

if __name__ == "__main__":
    main()
