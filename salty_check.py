"""
SALTY Data Integrity Checker (read-only)
Validates completeness and health of a salty_data/ download.

Usage:
    uv run salty_check.py salty_data
    uv run salty_check.py salty_data --source-csv 100k-205k_data.csv
    uv run salty_check.py /mnt/vol/salty_data --source-csv 0-100k_data.csv
    uv run salty_check.py salty_data --workers 8
"""

import argparse
import json
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from math import asin, cos, radians, sin, sqrt
from pathlib import Path

from PIL import Image
import numpy as np
import pandas as pd
from tqdm import tqdm

EXPECTED_IMAGES = {"000.jpg", "090.jpg", "180.jpg", "270.jpg"}
REQUIRED_META_FIELDS = {
    "index", "panoid", "pano_lat", "pano_lon",
    "original_lat", "original_lon",
    "headings", "view_resolution", "view_fov",
}

# File size thresholds for 1024x1024 JPEG at quality 90
MIN_FILE_SIZE = 5_000       # 5 KB — anything below this is suspicious
MAX_FILE_SIZE = 2_000_000   # 2 MB — anything above this is suspicious

# Blank image detection: flag if per-channel std deviation is below this
BLANK_STD_THRESHOLD = 5.0   # pixel values 0-255; uniform images ≈ 0

# Pano-to-original distance: flag if panorama is farther than this from request
MAX_PANO_DISTANCE_M = 500   # meters

# Expected metadata config values (must match scraper settings)
EXPECTED_HEADINGS = [0, 90, 180, 270]
EXPECTED_VIEW_RESOLUTION = "1024x1024"
EXPECTED_VIEW_FOV = 90.0

# California bounding box (with small margin for coastal/border panoramas)
CA_LAT_MIN, CA_LAT_MAX = 32.3, 42.1
CA_LON_MIN, CA_LON_MAX = -124.6, -114.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def haversine_m(lat1, lon1, lat2, lon2):
    """Great-circle distance between two points in meters."""
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return 6_371_000 * 2 * asin(sqrt(a))


def load_csv_indices(files):
    """Load index values from one or more CSVs. Returns (unique set, raw list)."""
    raw = []
    for f in files:
        try:
            raw.extend(pd.read_csv(f)["index"].values.tolist())
        except Exception as e:
            print(f"  WARNING: Could not read {f.name}: {e}")
    return set(raw), raw


def load_source_coords(source_path):
    """Load source CSV and return {index: (lat, lon)} dict."""
    df = pd.read_csv(source_path)
    idx_col = df.iloc[:, 0].astype(int)
    lat_col = df.iloc[:, 1].astype(float)
    lon_col = df.iloc[:, 2].astype(float)
    return dict(zip(idx_col, zip(lat_col, lon_col)))


def load_reject_reasons(files):
    """Load rejection reasons from rejects CSVs. Returns Counter of reasons."""
    reasons = Counter()
    for f in files:
        try:
            df = pd.read_csv(f)
            if "reason" in df.columns:
                reasons.update(df["reason"].dropna().values.tolist())
        except Exception:
            pass
    return reasons


def sample_list(items, n=20):
    """Return sorted first-n items for display."""
    return sorted(items)[:n]


# ---------------------------------------------------------------------------
# Per-folder worker (runs in thread pool)
# ---------------------------------------------------------------------------

def _process_folder(folder, has_metadata_dir, metadata_dir, source_coords):
    """Process one location folder. Returns a findings dict, or None if not a valid index folder."""
    try:
        idx = int(folder.name)
    except ValueError:
        return None

    result = dict(
        idx=idx,
        empty=False,
        incomplete=None,
        corrupt_imgs=[],
        bad_dimensions=[],
        bad_color_mode=[],
        size_outliers=[],
        blank_imgs=[],
        images_ok=0,
        missing_meta=False,
        corrupt_meta=None,
        meta_field_issues=None,
        meta_value_issues=[],
        meta_index_mismatch=None,
        coord_mismatch=None,
        pano_distance_issue=None,
        bad_coords=[],
        outside_california=[],
        panoid=None,
    )

    # --- [1] Image completeness ---
    present = {f.name for f in folder.glob("*.jpg")}
    missing = EXPECTED_IMAGES - present
    if not present:
        result["empty"] = True
    elif missing:
        result["incomplete"] = (idx, sorted(missing))

    # --- [2] Image integrity ---
    for img_path in sorted(folder.glob("*.jpg")):
        file_size = img_path.stat().st_size
        if file_size == 0:
            result["corrupt_imgs"].append((idx, img_path.name, "empty file (0 bytes)"))
            continue

        if file_size < MIN_FILE_SIZE:
            result["size_outliers"].append((idx, img_path.name, file_size, "suspiciously small"))
        elif file_size > MAX_FILE_SIZE:
            result["size_outliers"].append((idx, img_path.name, file_size, "suspiciously large"))

        try:
            with Image.open(img_path) as img:
                img.load()
                w, h = img.size
                if (w, h) != (1024, 1024):
                    result["bad_dimensions"].append((idx, img_path.name, f"{w}x{h}"))
                else:
                    result["images_ok"] += 1
                if img.mode != "RGB":
                    result["bad_color_mode"].append((idx, img_path.name, img.mode))
                std_val = float(np.array(img, dtype=np.float32).std())
                if std_val < BLANK_STD_THRESHOLD:
                    result["blank_imgs"].append((idx, img_path.name, round(std_val, 2)))
        except Exception as e:
            result["corrupt_imgs"].append((idx, img_path.name, str(e)))

    # --- [3] Metadata integrity ---
    if has_metadata_dir:
        meta_path = metadata_dir / f"{idx:06d}.json"
        if not meta_path.exists():
            result["missing_meta"] = True
        else:
            try:
                with open(meta_path) as f:
                    meta = json.load(f)

                missing_fields = REQUIRED_META_FIELDS - set(meta.keys())
                if missing_fields:
                    result["meta_field_issues"] = (idx, sorted(missing_fields))

                if "headings" in meta and meta["headings"] != EXPECTED_HEADINGS:
                    result["meta_value_issues"].append((idx, "headings", EXPECTED_HEADINGS, meta["headings"]))
                if "view_resolution" in meta and meta["view_resolution"] != EXPECTED_VIEW_RESOLUTION:
                    result["meta_value_issues"].append((idx, "view_resolution", EXPECTED_VIEW_RESOLUTION, meta["view_resolution"]))
                if "view_fov" in meta and meta["view_fov"] is not None:
                    try:
                        if float(meta["view_fov"]) != EXPECTED_VIEW_FOV:
                            result["meta_value_issues"].append((idx, "view_fov", EXPECTED_VIEW_FOV, meta["view_fov"]))
                    except (ValueError, TypeError):
                        result["meta_value_issues"].append((idx, "view_fov", EXPECTED_VIEW_FOV, meta["view_fov"]))

                if "index" in meta and int(meta["index"]) != idx:
                    result["meta_index_mismatch"] = (idx, meta["index"])

                if "panoid" in meta and meta["panoid"]:
                    result["panoid"] = meta["panoid"]

                if source_coords and idx in source_coords:
                    src_lat, src_lon = source_coords[idx]
                    meta_lat = meta.get("original_lat")
                    meta_lon = meta.get("original_lon")
                    if meta_lat is not None and meta_lon is not None:
                        if (abs(float(meta_lat) - src_lat) > 0.001
                                or abs(float(meta_lon) - src_lon) > 0.001):
                            result["coord_mismatch"] = (
                                idx, float(meta_lat), float(meta_lon), src_lat, src_lon,
                            )

                pano_lat = meta.get("pano_lat")
                pano_lon = meta.get("pano_lon")
                orig_lat = meta.get("original_lat")
                orig_lon = meta.get("original_lon")
                if all(v is not None for v in [pano_lat, pano_lon, orig_lat, orig_lon]):
                    try:
                        dist = haversine_m(
                            float(orig_lat), float(orig_lon),
                            float(pano_lat), float(pano_lon),
                        )
                        if dist > MAX_PANO_DISTANCE_M:
                            result["pano_distance_issue"] = (idx, round(dist, 1))
                    except (ValueError, TypeError):
                        pass

                for field, lat_key, lon_key in [
                    ("original", "original_lat", "original_lon"),
                    ("pano", "pano_lat", "pano_lon"),
                ]:
                    lat_v = meta.get(lat_key)
                    lon_v = meta.get(lon_key)
                    if lat_v is not None and lon_v is not None:
                        try:
                            la, lo = float(lat_v), float(lon_v)
                            if not (-90 <= la <= 90) or not (-180 <= lo <= 180):
                                result["bad_coords"].append((idx, field, la, lo))
                            elif not (CA_LAT_MIN <= la <= CA_LAT_MAX
                                      and CA_LON_MIN <= lo <= CA_LON_MAX):
                                result["outside_california"].append((idx, field, la, lo))
                        except (ValueError, TypeError):
                            result["bad_coords"].append((idx, field, lat_v, lon_v))

            except Exception as e:
                result["corrupt_meta"] = (idx, str(e))

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t0 = time.time()

    parser = argparse.ArgumentParser(description="SALTY data integrity checker")
    parser.add_argument(
        "data_dir",
        help="Path to salty_data directory",
    )
    parser.add_argument(
        "--source-csv",
        default=None,
        help="Path to source coordinate CSV (e.g. 100k-205k_data.csv) for coverage check",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel scan workers (default: 4)",
    )
    args = parser.parse_args()

    SALTY_DATA = Path(args.data_dir)
    IMAGES_DIR = SALTY_DATA / "images"
    METADATA_DIR = SALTY_DATA / "metadata"

    print("SALTY Integrity Check")
    print("=" * 60)
    print(f"Directory: {SALTY_DATA.resolve()}")

    if not SALTY_DATA.exists():
        print(f"ERROR: {SALTY_DATA} not found")
        sys.exit(1)
    if not IMAGES_DIR.exists():
        print(f"ERROR: {IMAGES_DIR} not found")
        sys.exit(1)

    has_metadata_dir = METADATA_DIR.exists()
    if not has_metadata_dir:
        print(f"WARNING: {METADATA_DIR} not found — metadata checks will be skipped")

    # ------------------------------------------------------------------
    # Load CSVs
    # ------------------------------------------------------------------
    print()
    print("Loading CSVs...")

    completed_files = sorted(SALTY_DATA.glob("completed*.csv"))
    rejected_files = sorted(SALTY_DATA.glob("rejects*.csv"))

    completed_set, completed_raw = set(), []
    if completed_files:
        completed_set, completed_raw = load_csv_indices(completed_files)
        names = ", ".join(f.name for f in completed_files)
        print(f"  completed : {names}")
        print(f"            → {len(completed_set):,} unique entries")
    else:
        print("  WARNING: No completed*.csv found")

    rejected_set, rejected_raw = set(), []
    reject_reasons = Counter()
    if rejected_files:
        rejected_set, rejected_raw = load_csv_indices(rejected_files)
        reject_reasons = load_reject_reasons(rejected_files)
        names = ", ".join(f.name for f in rejected_files)
        print(f"  rejects   : {names}")
        print(f"            → {len(rejected_set):,} unique entries")
    else:
        print("  WARNING: No rejects*.csv found")

    source_coords = None
    source_indices = None
    if args.source_csv:
        source_path = Path(args.source_csv)
        if not source_path.exists():
            print(f"  WARNING: --source-csv {source_path} not found, skipping coverage check")
        else:
            source_coords = load_source_coords(source_path)
            source_indices = set(source_coords.keys())
            print(f"  source    : {source_path.name} → {len(source_indices):,} entries")

    # ------------------------------------------------------------------
    # Collect folders (single pass for disk_indices + folder list)
    # ------------------------------------------------------------------
    folders = []
    disk_indices = set()
    for entry in IMAGES_DIR.iterdir():
        if entry.is_dir():
            try:
                disk_indices.add(int(entry.name))
                folders.append(entry)
            except ValueError:
                pass
    folders.sort(key=lambda f: int(f.name))

    print(f"\nScanning {len(folders):,} image folders with {args.workers} workers...")

    # Accumulators
    incomplete = []
    empty_folders = []
    corrupt_imgs = []
    bad_dimensions = []
    bad_color_mode = []
    size_outliers = []
    blank_imgs = []
    missing_meta = []
    corrupt_meta = []
    meta_field_issues = []
    meta_value_issues = []
    meta_index_mismatch = []
    coord_mismatches = []
    pano_distance_issues = []
    bad_coords = []
    outside_california = []
    panoid_map = {}
    total_images_ok = 0

    worker = partial(
        _process_folder,
        has_metadata_dir=has_metadata_dir,
        metadata_dir=METADATA_DIR,
        source_coords=source_coords,
    )

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(worker, folder): folder for folder in folders}
        for future in tqdm(as_completed(futures), total=len(folders), desc="Scanning", unit="loc"):
            r = future.result()
            if r is None:
                continue
            if r["empty"]:
                empty_folders.append(r["idx"])
            if r["incomplete"]:
                incomplete.append(r["incomplete"])
            corrupt_imgs.extend(r["corrupt_imgs"])
            bad_dimensions.extend(r["bad_dimensions"])
            bad_color_mode.extend(r["bad_color_mode"])
            size_outliers.extend(r["size_outliers"])
            blank_imgs.extend(r["blank_imgs"])
            total_images_ok += r["images_ok"]
            if r["missing_meta"]:
                missing_meta.append(r["idx"])
            if r["corrupt_meta"]:
                corrupt_meta.append(r["corrupt_meta"])
            if r["meta_field_issues"]:
                meta_field_issues.append(r["meta_field_issues"])
            meta_value_issues.extend(r["meta_value_issues"])
            if r["meta_index_mismatch"]:
                meta_index_mismatch.append(r["meta_index_mismatch"])
            if r["coord_mismatch"]:
                coord_mismatches.append(r["coord_mismatch"])
            if r["pano_distance_issue"]:
                pano_distance_issues.append(r["pano_distance_issue"])
            bad_coords.extend(r["bad_coords"])
            outside_california.extend(r["outside_california"])
            if r["panoid"]:
                panoid_map.setdefault(r["panoid"], []).append(r["idx"])

    # ------------------------------------------------------------------
    # [4] Source CSV coverage
    # ------------------------------------------------------------------
    never_tried = set()
    if source_indices is not None:
        attempted = completed_set | rejected_set
        never_tried = source_indices - attempted

    # ------------------------------------------------------------------
    # [5] Completed ↔ images consistency
    # ------------------------------------------------------------------
    folders_not_in_completed = disk_indices - completed_set
    completed_without_folder = completed_set - disk_indices

    # ------------------------------------------------------------------
    # [6] Additional checks
    # ------------------------------------------------------------------
    completed_counts = Counter(completed_raw)
    duplicate_completed = {idx: n for idx, n in completed_counts.items() if n > 1}

    rejected_counts = Counter(rejected_raw)
    duplicate_rejected = {idx: n for idx, n in rejected_counts.items() if n > 1}

    in_both = completed_set & rejected_set

    orphan_meta = set()
    if has_metadata_dir:
        for meta_file in METADATA_DIR.iterdir():
            if meta_file.suffix == ".json":
                try:
                    meta_idx = int(meta_file.stem)
                    if meta_idx not in disk_indices:
                        orphan_meta.add(meta_idx)
                except ValueError:
                    pass

    duplicate_panoids = {pid: idxs for pid, idxs in panoid_map.items() if len(idxs) > 1}

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    print()
    print("Results")
    print("=" * 60)

    issues = 0

    # [1]
    n1_ok = len(disk_indices) - len(incomplete) - len(empty_folders)
    n1_bad = len(incomplete) + len(empty_folders)
    issues += n1_bad
    tag1 = "PASS" if n1_bad == 0 else "FAIL"
    parts1 = f"{n1_ok:,} OK / {len(incomplete):,} incomplete"
    if empty_folders:
        parts1 += f" / {len(empty_folders):,} empty"
    print(f"[1] Image completeness (4 per folder)    : {tag1} — {parts1}")

    # [2]
    n2_bad = len(corrupt_imgs) + len(bad_dimensions) + len(bad_color_mode) + len(blank_imgs)
    issues += n2_bad
    tag2 = "PASS" if n2_bad == 0 else "FAIL"
    parts2 = f"{total_images_ok:,} OK / {len(corrupt_imgs):,} corrupt / {len(bad_dimensions):,} wrong size"
    if bad_color_mode:
        parts2 += f" / {len(bad_color_mode):,} non-RGB"
    if blank_imgs:
        parts2 += f" / {len(blank_imgs):,} blank/degenerate"
    if size_outliers:
        parts2 += f" / {len(size_outliers):,} size outliers"
    print(f"[2] Image integrity (readable, 1024x1024): {tag2} — {parts2}")

    # [3]
    if has_metadata_dir:
        n3_bad = (len(missing_meta) + len(corrupt_meta) + len(meta_field_issues)
                  + len(meta_value_issues) + len(meta_index_mismatch)
                  + len(coord_mismatches) + len(pano_distance_issues)
                  + len(bad_coords) + len(outside_california))
        issues += n3_bad
        n3_ok = len(disk_indices) - len(missing_meta) - len(corrupt_meta)
        tag3 = "PASS" if n3_bad == 0 else "FAIL"
        parts3 = f"{n3_ok:,} OK / {len(missing_meta):,} missing / {len(corrupt_meta):,} corrupt"
        if meta_field_issues:
            parts3 += f" / {len(meta_field_issues):,} incomplete fields"
        if meta_value_issues:
            parts3 += f" / {len(meta_value_issues):,} wrong config values"
        if meta_index_mismatch:
            parts3 += f" / {len(meta_index_mismatch):,} index mismatch"
        if coord_mismatches:
            parts3 += f" / {len(coord_mismatches):,} coordinate mismatch"
        if pano_distance_issues:
            parts3 += f" / {len(pano_distance_issues):,} pano too far (>{MAX_PANO_DISTANCE_M}m)"
        if bad_coords:
            parts3 += f" / {len(bad_coords):,} out-of-bounds coords"
        if outside_california:
            parts3 += f" / {len(outside_california):,} outside California"
        print(f"[3] Metadata integrity                   : {tag3} — {parts3}")
    else:
        print(f"[3] Metadata integrity                   : SKIP — metadata/ not found")

    # [4]
    if source_indices is not None:
        n4_attempted = len(source_indices) - len(never_tried)
        tag4 = "PASS" if len(never_tried) == 0 else "WARN"
        print(f"[4] Source CSV coverage                   : {tag4} — {n4_attempted:,} / {len(source_indices):,} attempted, {len(never_tried):,} never tried")
    else:
        print(f"[4] Source CSV coverage                   : SKIP — no --source-csv provided")

    # [5]
    n5_bad = len(folders_not_in_completed)
    issues += n5_bad
    tag5 = "PASS" if n5_bad == 0 else "FAIL"
    print(f"[5] Completed <-> images match            : {tag5} — {len(folders_not_in_completed):,} folders not in completed / {len(completed_without_folder):,} completed without folder")

    # [6]
    n6_bad = len(duplicate_completed) + len(in_both) + len(duplicate_panoids)
    issues += n6_bad
    print(f"[6] Additional")
    print(f"    Duplicate completed entries           : {len(duplicate_completed):,}")
    print(f"    Duplicate rejected entries            : {len(duplicate_rejected):,}")
    print(f"    In BOTH completed AND rejected        : {len(in_both):,}")
    print(f"    Duplicate panoid (same pano, diff loc): {len(duplicate_panoids):,}")
    if has_metadata_dir:
        print(f"    Orphan metadata (no image folder)     : {len(orphan_meta):,}")
    if reject_reasons:
        print(f"    Reject reason breakdown:")
        for reason, count in reject_reasons.most_common():
            print(f"      {reason:40s} {count:,}")

    # ------------------------------------------------------------------
    # Details (first N of each failure)
    # ------------------------------------------------------------------
    details = []

    if empty_folders:
        lines = ["", f"Empty folders — 0 images (first 20 of {len(empty_folders):,}):"]
        for idx in sample_list(empty_folders)[:20]:
            lines.append(f"  {idx:06d}/")
        details.extend(lines)

    if incomplete:
        lines = ["", f"Incomplete folders (first 20 of {len(incomplete):,}):"]
        for idx, missing in sample_list(incomplete)[:20]:
            lines.append(f"  {idx:06d}  missing: {missing}")
        details.extend(lines)

    if corrupt_imgs:
        lines = ["", f"Corrupt images (first 20 of {len(corrupt_imgs):,}):"]
        for idx, fname, err in corrupt_imgs[:20]:
            lines.append(f"  {idx:06d}/{fname}  {err}")
        details.extend(lines)

    if bad_dimensions:
        lines = ["", f"Wrong dimensions (first 20 of {len(bad_dimensions):,}):"]
        for idx, fname, dims in bad_dimensions[:20]:
            lines.append(f"  {idx:06d}/{fname}  {dims} (expected 1024x1024)")
        details.extend(lines)

    if size_outliers:
        lines = ["", f"File size outliers (first 20 of {len(size_outliers):,}):"]
        for idx, fname, sz, reason in size_outliers[:20]:
            lines.append(f"  {idx:06d}/{fname}  {sz:,} bytes — {reason}")
        details.extend(lines)

    if bad_color_mode:
        lines = ["", f"Non-RGB images (first 20 of {len(bad_color_mode):,}):"]
        for idx, fname, mode in bad_color_mode[:20]:
            lines.append(f"  {idx:06d}/{fname}  mode={mode} (expected RGB)")
        details.extend(lines)

    if blank_imgs:
        lines = ["", f"Blank/degenerate images — std < {BLANK_STD_THRESHOLD} (first 20 of {len(blank_imgs):,}):"]
        for idx, fname, std_val in blank_imgs[:20]:
            lines.append(f"  {idx:06d}/{fname}  std={std_val}")
        details.extend(lines)

    if missing_meta:
        lines = ["", f"Missing metadata (first 20 of {len(missing_meta):,}):"]
        for idx in sample_list(missing_meta)[:20]:
            lines.append(f"  {idx:06d}.json")
        details.extend(lines)

    if corrupt_meta:
        lines = ["", f"Corrupt metadata (first 20 of {len(corrupt_meta):,}):"]
        for idx, err in corrupt_meta[:20]:
            lines.append(f"  {idx:06d}.json  {err}")
        details.extend(lines)

    if meta_field_issues:
        lines = ["", f"Metadata missing required fields (first 20 of {len(meta_field_issues):,}):"]
        for idx, fields in meta_field_issues[:20]:
            lines.append(f"  {idx:06d}.json  missing: {fields}")
        details.extend(lines)

    if meta_value_issues:
        lines = ["", f"Metadata wrong config values (first 20 of {len(meta_value_issues):,}):"]
        for idx, field, expected, actual in meta_value_issues[:20]:
            lines.append(f"  {idx:06d}.json  {field}: expected {expected}, got {actual}")
        details.extend(lines)

    if meta_index_mismatch:
        lines = ["", f"Metadata index mismatch (first 20 of {len(meta_index_mismatch):,}):"]
        for idx, json_idx in meta_index_mismatch[:20]:
            lines.append(f"  folder {idx:06d} but JSON has index={json_idx}")
        details.extend(lines)

    if coord_mismatches:
        lines = ["", f"Coordinate mismatch: metadata vs source CSV (first 20 of {len(coord_mismatches):,}):"]
        for idx, mlat, mlon, slat, slon in coord_mismatches[:20]:
            lines.append(f"  {idx:06d}  meta=({mlat:.6f}, {mlon:.6f})  source=({slat:.6f}, {slon:.6f})")
        details.extend(lines)

    if pano_distance_issues:
        lines = ["", f"Panorama too far from original location (>{MAX_PANO_DISTANCE_M}m, first 20 of {len(pano_distance_issues):,}):"]
        for idx, dist in sorted(pano_distance_issues, key=lambda x: -x[1])[:20]:
            lines.append(f"  {idx:06d}  {dist:,.0f}m away")
        details.extend(lines)

    if bad_coords:
        lines = ["", f"Out-of-bounds coordinates (first 20 of {len(bad_coords):,}):"]
        for idx, field, la, lo in bad_coords[:20]:
            lines.append(f"  {idx:06d}  {field}: lat={la}, lon={lo}")
        details.extend(lines)

    if outside_california:
        lines = ["", f"Coordinates outside California bbox (first 20 of {len(outside_california):,}):"]
        for idx, field, la, lo in outside_california[:20]:
            lines.append(f"  {idx:06d}  {field}: ({la:.6f}, {lo:.6f})")
        details.extend(lines)

    if duplicate_panoids:
        lines = ["", f"Duplicate panoids — same panorama used by multiple locations (first 20 of {len(duplicate_panoids):,}):"]
        for pid in sample_list(duplicate_panoids)[:20]:
            idxs = duplicate_panoids[pid]
            lines.append(f"  {pid}  → indices {idxs[:5]}{'...' if len(idxs) > 5 else ''}")
        details.extend(lines)

    if never_tried:
        lines = ["", f"Never attempted indices (first 20 of {len(never_tried):,}):"]
        for idx in sample_list(never_tried)[:20]:
            lines.append(f"  {idx}")
        details.extend(lines)

    if folders_not_in_completed:
        lines = ["", f"Image folders not in any completed CSV (first 20 of {len(folders_not_in_completed):,}):"]
        for idx in sample_list(folders_not_in_completed)[:20]:
            lines.append(f"  {idx:06d}")
        details.extend(lines)

    if duplicate_completed:
        lines = ["", f"Duplicate completed entries (first 20 of {len(duplicate_completed):,}):"]
        for idx in sample_list(duplicate_completed)[:20]:
            lines.append(f"  index {idx} appears {duplicate_completed[idx]}x")
        details.extend(lines)

    if in_both:
        lines = ["", f"In BOTH completed and rejected (first 20 of {len(in_both):,}):"]
        for idx in sample_list(in_both)[:20]:
            lines.append(f"  {idx}")
        details.extend(lines)

    if details:
        print("\n" + "-" * 60)
        print("Details")
        print("-" * 60)
        for line in details:
            print(line)

    # ------------------------------------------------------------------
    # Verdict
    # ------------------------------------------------------------------
    elapsed = time.time() - t0
    print()
    print("=" * 60)
    if issues == 0:
        print("All checks passed.")
    else:
        print(f"{issues:,} issue(s) found.")
    print(f"Completed in {elapsed:.1f}s")
    print()


if __name__ == "__main__":
    main()
