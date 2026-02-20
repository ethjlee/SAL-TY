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
import hashlib
import io
import json
import os
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import date
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from math import asin, cos, radians, sin, sqrt
from pathlib import Path

_DATE_RE = re.compile(r"^\d{4}-\d{2}$")

from PIL import Image
import numpy as np
import pandas as pd
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

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

# Blur detection: Laplacian variance of grayscale — high = sharp, low = blurry
BLUR_THRESHOLD = 10.0       # zoom 3 / Q90 street-view typically scores >30; <10 is visibly blurry

# Pano-to-original distance: flag if panorama is farther than this from request
MAX_PANO_DISTANCE_M = 500   # meters

# Expected metadata config values (must match scraper settings)
EXPECTED_HEADINGS = [0, 90, 180, 270]
EXPECTED_VIEW_RESOLUTION = "1024x1024"
EXPECTED_VIEW_FOV = 90.0

# California bounding box (with small margin for coastal/border panoramas)
CA_LAT_MIN, CA_LAT_MAX = 32.3, 42.1
CA_LON_MIN, CA_LON_MAX = -124.6, -114.0

# Coordinate comparison tolerance — ~111m at equator
COORD_TOLERANCE = 0.001

# ITU-R BT.601 luminance weights for fast RGB → grayscale (avoids second PIL decode)
_BT601_R, _BT601_G, _BT601_B = 0.299, 0.587, 0.114

# JPEG end-of-image marker for truncation check
_JPEG_EOI = b"\xff\xd9"


# ---------------------------------------------------------------------------
# Findings containers
# ---------------------------------------------------------------------------

@dataclass
class ScanFindings:
    """Accumulated results from the parallel per-folder scan."""
    empty_folders:         list = field(default_factory=list)   # list[int]
    incomplete:            list = field(default_factory=list)   # list[tuple[int, list[str]]]
    corrupt_imgs:          list = field(default_factory=list)   # list[tuple[int, str, str]]
    bad_dimensions:        list = field(default_factory=list)   # list[tuple[int, str, str]]
    bad_color_mode:        list = field(default_factory=list)   # list[tuple[int, str, str]]
    size_outliers:         list = field(default_factory=list)   # list[tuple[int, str, int, str]]
    blank_imgs:            list = field(default_factory=list)   # list[tuple[int, str, float]]
    blurry_imgs:           list = field(default_factory=list)   # list[tuple[int, str, float]]
    truncated_imgs:        list = field(default_factory=list)   # list[tuple[int, str]]
    duplicate_views:       list = field(default_factory=list)   # list[tuple[int, str, str]]
    unexpected_files:      list = field(default_factory=list)   # list[tuple[int, str]]
    missing_meta:          list = field(default_factory=list)   # list[int]
    corrupt_meta:          list = field(default_factory=list)   # list[tuple[int, str]]
    meta_field_issues:     list = field(default_factory=list)   # list[tuple[int, list[str]]]
    meta_value_issues:     list = field(default_factory=list)   # list[tuple[int, str, Any, Any]]
    meta_index_mismatch:   list = field(default_factory=list)   # list[tuple[int, Any]]
    coord_mismatches:      list = field(default_factory=list)   # list[tuple[int, f, f, f, f]]
    pano_distance_issues:  list = field(default_factory=list)   # list[tuple[int, float]]
    bad_coords:            list = field(default_factory=list)   # list[tuple[int, str, Any, Any]]
    outside_california:    list = field(default_factory=list)   # list[tuple[int, str, float, float]]
    copyright_issues:      list = field(default_factory=list)   # list[tuple[int, str]]
    country_code_issues:   list = field(default_factory=list)   # list[tuple[int, str]]
    bad_dates:             list = field(default_factory=list)   # list[tuple[int, str]]
    panoid_csv_mismatches: list = field(default_factory=list)   # list[tuple[int, str, str]]
    panoid_map:            dict = field(default_factory=dict)   # dict[str, list[int]]
    total_images_ok:       int  = 0
    total_disk_bytes:      int  = 0


@dataclass
class DerivedFindings:
    """Post-scan derived checks (set arithmetic, Counter analysis)."""
    never_tried:              set  = field(default_factory=set)
    folders_not_in_completed: set  = field(default_factory=set)
    completed_without_folder: set  = field(default_factory=set)
    duplicate_completed:      dict = field(default_factory=dict)  # dict[int, int]
    duplicate_rejected:       dict = field(default_factory=dict)  # dict[int, int]
    in_both:                  set  = field(default_factory=set)
    orphan_meta:              set  = field(default_factory=set)
    duplicate_panoids:        dict = field(default_factory=dict)  # dict[str, list[int]]


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
    for csv_file in files:
        try:
            series = pd.read_csv(csv_file)["index"]
            bad = pd.to_numeric(series, errors="coerce").isna()
            if bad.any():
                print(f"  WARNING: {bad.sum()} non-integer value(s) in index column of {csv_file.name} — skipping those rows")
            raw.extend(pd.to_numeric(series, errors="coerce").dropna().astype(int).values.tolist())
        except Exception as e:
            print(f"  WARNING: Could not read {csv_file.name}: {e}")
    return set(raw), raw


def load_completed_panoids(files):
    """Load {index: panoid} from completed CSVs. Used for CSV↔metadata cross-check."""
    result = {}
    for csv_file in files:
        try:
            df = pd.read_csv(csv_file)
            if "panoid" in df.columns and "index" in df.columns:
                sub = df[["index", "panoid"]].dropna(subset=["panoid"])
                sub = sub.astype({"index": int, "panoid": str})
                result.update(zip(sub["index"], sub["panoid"]))
        except Exception as e:
            print(f"  WARNING: Could not read panoids from {csv_file.name}: {e}")
    return result


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
    for csv_file in files:
        try:
            df = pd.read_csv(csv_file)
            if "reason" in df.columns:
                reasons.update(df["reason"].dropna().values.tolist())
        except Exception:
            pass
    return reasons


def write_flagged_export(path, sections):
    """Write flagged indices to a review file. Returns count of entries written."""
    total = 0
    lines = [
        f"# SALTY flagged locations — {date.today()}",
        "# DELETE lines you want to KEEP in completed.csv.",
        "# Lines you leave will be rejected by salty_reject.py.",
        "# Run: uv run salty_reject.py <data_dir> --from-file flagged.txt",
    ]
    for label, entries in sections:
        if not entries:
            continue
        lines.append("#")
        lines.append(f"# --- {label} ({len(entries):,}) ---")
        for idx, detail in entries:
            lines.append(f"{idx:06d}  # {label}: {detail}")
            total += 1
    if total == 0:
        return 0
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return total


def _detail_block(header, items, formatter, n=20):
    """Return a list of lines for a details block, or [] if items is empty."""
    if not items:
        return []
    lst = list(items)
    total = len(lst)
    label = f"first {n} of {total:,}" if total > n else f"{total:,}"
    lines = ["", f"{header} ({label}):"]
    for item in lst[:n]:
        lines.append("  " + formatter(item))
    return lines


# ---------------------------------------------------------------------------
# Main-process helpers extracted from main()
# ---------------------------------------------------------------------------

def _parse_args():
    """Parse CLI arguments."""
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
        default=os.cpu_count(),
        help=f"Number of parallel scan workers (default: {os.cpu_count()})",
    )
    parser.add_argument(
        "--no-export",
        action="store_true",
        help="Do not write flagged.txt (default: always write)",
    )
    return parser.parse_args()


def _validate_dirs(salty_data, images_dir):
    """Validate required directories exist. Returns has_metadata_dir flag."""
    if not salty_data.exists():
        print(f"ERROR: {salty_data} not found")
        sys.exit(1)
    if not images_dir.exists():
        print(f"ERROR: {images_dir} not found")
        sys.exit(1)
    metadata_dir = salty_data / "metadata"
    has_metadata_dir = metadata_dir.exists()
    if not has_metadata_dir:
        print(f"WARNING: {metadata_dir} not found — metadata checks will be skipped")
    return has_metadata_dir


def _load_csvs(salty_data):
    """Discover and load completed + rejected CSVs. Returns dict of all derived data."""
    completed_files = sorted(salty_data.glob("completed*.csv"))
    rejected_files  = sorted(salty_data.glob("rejects*.csv"))

    completed_set, completed_raw = set(), []
    completed_panoids = {}
    if completed_files:
        completed_set, completed_raw = load_csv_indices(completed_files)
        completed_panoids = load_completed_panoids(completed_files)
        names = ", ".join(csv_file.name for csv_file in completed_files)
        print(f"  completed : {names}")
        print(f"            → {len(completed_set):,} unique entries")
    else:
        print("  WARNING: No completed*.csv found")

    rejected_set, rejected_raw = set(), []
    reject_reasons = Counter()
    if rejected_files:
        rejected_set, rejected_raw = load_csv_indices(rejected_files)
        reject_reasons = load_reject_reasons(rejected_files)
        names = ", ".join(csv_file.name for csv_file in rejected_files)
        print(f"  rejects   : {names}")
        print(f"            → {len(rejected_set):,} unique entries")
    else:
        print("  WARNING: No rejects*.csv found")

    return dict(
        completed_set=completed_set,
        completed_raw=completed_raw,
        completed_panoids=completed_panoids,
        rejected_set=rejected_set,
        rejected_raw=rejected_raw,
        reject_reasons=reject_reasons,
    )


def _load_source_csv(source_path):
    """Load source coordinate CSV. Returns (source_coords, source_indices) or (None, None)."""
    if not source_path.exists():
        print(f"  WARNING: --source-csv {source_path} not found, skipping coverage check")
        return None, None
    source_coords  = load_source_coords(source_path)
    source_indices = set(source_coords.keys())
    print(f"  source    : {source_path.name} → {len(source_indices):,} entries")
    return source_coords, source_indices


def _collect_folders(images_dir):
    """Enumerate numeric subdirectories of images_dir. Returns (folders, disk_indices)."""
    folders = []
    disk_indices = set()
    for entry in images_dir.iterdir():
        if entry.is_dir():
            try:
                disk_indices.add(int(entry.name))
                folders.append(entry)
            except ValueError:
                pass
    folders.sort(key=lambda f: int(f.name))
    return folders, disk_indices


def _scan_all_folders(folders, has_metadata_dir, metadata_dir, source_coords, completed_panoids, n_workers):
    """Run parallel folder scan. Returns consolidated ScanFindings."""
    findings = ScanFindings()

    worker = partial(_process_folder, has_metadata_dir=has_metadata_dir, metadata_dir=metadata_dir)

    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_init_worker,
        initargs=(source_coords, completed_panoids),
    ) as executor:
        futures = [executor.submit(worker, folder) for folder in folders]
        for future in tqdm(
            as_completed(futures),
            total=len(folders), desc="Scanning", unit="loc", smoothing=0.3,
        ):
            folder_result = future.result()
            if folder_result is None:
                continue

            # Boolean flags → list entries
            if folder_result["empty"]:
                findings.empty_folders.append(folder_result["idx"])
            if folder_result["incomplete"]:
                findings.incomplete.append(folder_result["incomplete"])
            if folder_result["missing_meta"]:
                findings.missing_meta.append(folder_result["idx"])
            if folder_result["corrupt_meta"]:
                findings.corrupt_meta.append(folder_result["corrupt_meta"])
            if folder_result["meta_field_issues"]:
                findings.meta_field_issues.append(folder_result["meta_field_issues"])
            if folder_result["meta_index_mismatch"]:
                findings.meta_index_mismatch.append(folder_result["meta_index_mismatch"])
            if folder_result["coord_mismatch"]:
                findings.coord_mismatches.append(folder_result["coord_mismatch"])
            if folder_result["pano_distance_issue"]:
                findings.pano_distance_issues.append(folder_result["pano_distance_issue"])
            if folder_result["panoid"]:
                findings.panoid_map.setdefault(folder_result["panoid"], []).append(folder_result["idx"])

            # Numeric aggregates
            findings.total_images_ok  += folder_result["images_ok"]
            findings.total_disk_bytes += folder_result["disk_bytes"]

            # List fields — extend directly
            findings.corrupt_imgs.extend(folder_result["corrupt_imgs"])
            findings.bad_dimensions.extend(folder_result["bad_dimensions"])
            findings.bad_color_mode.extend(folder_result["bad_color_mode"])
            findings.size_outliers.extend(folder_result["size_outliers"])
            findings.blank_imgs.extend(folder_result["blank_imgs"])
            findings.blurry_imgs.extend(folder_result["blurry_imgs"])
            findings.truncated_imgs.extend(folder_result["truncated_imgs"])
            findings.duplicate_views.extend(folder_result["duplicate_views"])
            findings.unexpected_files.extend(folder_result["unexpected_files"])
            findings.meta_value_issues.extend(folder_result["meta_value_issues"])
            findings.bad_coords.extend(folder_result["bad_coords"])
            findings.outside_california.extend(folder_result["outside_california"])
            findings.copyright_issues.extend(folder_result["copyright_issues"])
            findings.country_code_issues.extend(folder_result["country_code_issues"])
            findings.bad_dates.extend(folder_result["bad_dates"])
            findings.panoid_csv_mismatches.extend(folder_result["panoid_csv_mismatches"])

    return findings


def _compute_derived(findings, csv_data, disk_indices, has_metadata_dir, metadata_dir, source_indices):
    """Compute post-scan derived checks. Returns DerivedFindings."""
    derived = DerivedFindings()

    if source_indices is not None:
        attempted = csv_data["completed_set"] | csv_data["rejected_set"]
        derived.never_tried = source_indices - attempted

    derived.folders_not_in_completed = disk_indices - csv_data["completed_set"]
    derived.completed_without_folder = csv_data["completed_set"] - disk_indices

    completed_counts = Counter(csv_data["completed_raw"])
    derived.duplicate_completed = {idx: n for idx, n in completed_counts.items() if n > 1}

    rejected_counts = Counter(csv_data["rejected_raw"])
    derived.duplicate_rejected = {idx: n for idx, n in rejected_counts.items() if n > 1}

    derived.in_both = csv_data["completed_set"] & csv_data["rejected_set"]

    if has_metadata_dir:
        for meta_file in metadata_dir.iterdir():
            if meta_file.suffix == ".json":
                try:
                    meta_idx = int(meta_file.stem)
                    if meta_idx not in disk_indices:
                        derived.orphan_meta.add(meta_idx)
                except ValueError:
                    pass

    derived.duplicate_panoids = {
        pid: idxs for pid, idxs in findings.panoid_map.items() if len(idxs) > 1
    }

    return derived


# ---------------------------------------------------------------------------
# Report detail table — drives the Details section of the report
# Each entry: (header, items_fn(findings, derived), formatter(item))
# ---------------------------------------------------------------------------

_DETAIL_SPECS = [
    (
        "Empty folders — 0 images",
        lambda f, _: sorted(f.empty_folders),
        lambda x: f"{x:06d}/",
    ),
    (
        "Incomplete folders",
        lambda f, _: sorted(f.incomplete),
        lambda x: f"{x[0]:06d}  missing: {x[1]}",
    ),
    (
        "Corrupt images",
        lambda f, _: f.corrupt_imgs,
        lambda x: f"{x[0]:06d}/{x[1]}  {x[2]}",
    ),
    (
        "Truncated JPEGs — missing FFD9 end marker",
        lambda f, _: f.truncated_imgs,
        lambda x: f"{x[0]:06d}/{x[1]}",
    ),
    (
        "Duplicate views within location — identical file content",
        lambda f, _: f.duplicate_views,
        lambda x: f"{x[0]:06d}/{x[1]}  identical to {x[2]}",
    ),
    (
        "Wrong dimensions",
        lambda f, _: f.bad_dimensions,
        lambda x: f"{x[0]:06d}/{x[1]}  {x[2]} (expected 1024x1024)",
    ),
    (
        "File size outliers",
        lambda f, _: f.size_outliers,
        lambda x: f"{x[0]:06d}/{x[1]}  {x[2]:,} bytes — {x[3]}",
    ),
    (
        "Non-RGB images",
        lambda f, _: f.bad_color_mode,
        lambda x: f"{x[0]:06d}/{x[1]}  mode={x[2]} (expected RGB)",
    ),
    (
        f"Blank/degenerate images — std < {BLANK_STD_THRESHOLD}",
        lambda f, _: f.blank_imgs,
        lambda x: f"{x[0]:06d}/{x[1]}  std={x[2]}",
    ),
    (
        f"Blurry images — Laplacian variance < {BLUR_THRESHOLD}",
        lambda f, _: f.blurry_imgs,
        lambda x: f"{x[0]:06d}/{x[1]}  blur_score={x[2]}",
    ),
    (
        "Missing metadata",
        lambda f, _: sorted(f.missing_meta),
        lambda x: f"{x:06d}.json",
    ),
    (
        "Corrupt metadata",
        lambda f, _: f.corrupt_meta,
        lambda x: f"{x[0]:06d}.json  {x[1]}",
    ),
    (
        "Metadata missing required fields",
        lambda f, _: f.meta_field_issues,
        lambda x: f"{x[0]:06d}.json  missing: {x[1]}",
    ),
    (
        "Metadata wrong config values",
        lambda f, _: f.meta_value_issues,
        lambda x: f"{x[0]:06d}.json  {x[1]}: expected {x[2]}, got {x[3]}",
    ),
    (
        "Metadata index mismatch",
        lambda f, _: f.meta_index_mismatch,
        lambda x: f"folder {x[0]:06d} but JSON has index={x[1]}",
    ),
    (
        "Coordinate mismatch: metadata vs source CSV",
        lambda f, _: f.coord_mismatches,
        lambda x: f"{x[0]:06d}  meta=({x[1]:.6f}, {x[2]:.6f})  source=({x[3]:.6f}, {x[4]:.6f})",
    ),
    (
        f"Panorama too far from original location (>{MAX_PANO_DISTANCE_M}m)",
        lambda f, _: sorted(f.pano_distance_issues, key=lambda x: -x[1]),
        lambda x: f"{x[0]:06d}  {x[1]:,.0f}m away",
    ),
    (
        "Out-of-bounds coordinates",
        lambda f, _: f.bad_coords,
        lambda x: f"{x[0]:06d}  {x[1]}: lat={x[2]}, lon={x[3]}",
    ),
    (
        "Coordinates outside California bbox",
        lambda f, _: f.outside_california,
        lambda x: f"{x[0]:06d}  {x[1]}: ({x[2]:.6f}, {x[3]:.6f})",
    ),
    (
        "Non-Google copyright — possible photosphere or user content",
        lambda f, _: f.copyright_issues,
        lambda x: f"{x[0]:06d}  copyright={repr(x[1])}",
    ),
    (
        "Wrong country code — expected US",
        lambda f, _: f.country_code_issues,
        lambda x: f"{x[0]:06d}  country_code={repr(x[1])}",
    ),
    (
        "Bad capture date format — expected YYYY-MM",
        lambda f, _: f.bad_dates,
        lambda x: f"{x[0]:06d}  date={repr(x[1])}",
    ),
    (
        "Panoid mismatch between completed CSV and metadata JSON",
        lambda f, _: f.panoid_csv_mismatches,
        lambda x: f"{x[0]:06d}  csv={x[1]}  json={x[2]}",
    ),
    (
        "Duplicate panoids — same panorama used by multiple locations",
        lambda _, d: sorted(d.duplicate_panoids.items()),
        lambda x: f"{x[0]}  → indices {sorted(x[1])[:5]}{'...' if len(x[1]) > 5 else ''}",
    ),
    (
        "Unexpected files in image folders",
        lambda f, _: f.unexpected_files,
        lambda x: f"{x[0]:06d}/{x[1]}",
    ),
    (
        "Never attempted indices",
        lambda _, d: sorted(d.never_tried),
        lambda x: str(x),
    ),
    (
        "Image folders not in any completed CSV",
        lambda _, d: sorted(d.folders_not_in_completed),
        lambda x: f"{x:06d}",
    ),
    (
        "Completed entries without an image folder",
        lambda _, d: sorted(d.completed_without_folder),
        lambda x: f"{x:06d}",
    ),
    (
        "Duplicate completed entries",
        lambda _, d: sorted(d.duplicate_completed.items()),
        lambda x: f"index {x[0]} appears {x[1]}x",
    ),
    (
        "Duplicate rejected entries",
        lambda _, d: sorted(d.duplicate_rejected.items()),
        lambda x: f"index {x[0]} appears {x[1]}x",
    ),
    (
        "In BOTH completed and rejected",
        lambda _, d: sorted(d.in_both),
        lambda x: str(x),
    ),
    (
        "Orphan metadata — JSON with no image folder",
        lambda _, d: sorted(d.orphan_meta),
        lambda x: f"{x:06d}.json",
    ),
]


# ---------------------------------------------------------------------------
# Flagged export table — drives flagged.txt section building
# Each entry: (label, entries_fn(findings, derived) → list[tuple[int, str]])
# ---------------------------------------------------------------------------

_EXPORT_SPECS = [
    (
        "empty_folders",
        lambda f, _: [(idx, "empty folder") for idx in sorted(f.empty_folders)],
    ),
    (
        "incomplete",
        lambda f, _: [(x[0], f"missing: {x[1]}") for x in sorted(f.incomplete)],
    ),
    (
        "corrupt_imgs",
        lambda f, _: [(x[0], f"{x[1]}: {x[2]}") for x in f.corrupt_imgs],
    ),
    (
        "bad_dimensions",
        lambda f, _: [(x[0], f"{x[1]}: {x[2]} (expected 1024x1024)") for x in f.bad_dimensions],
    ),
    (
        "bad_color_mode",
        lambda f, _: [(x[0], f"{x[1]}: mode={x[2]} (expected RGB)") for x in f.bad_color_mode],
    ),
    (
        "size_outliers",
        lambda f, _: [(x[0], f"{x[1]}: {x[2]:,} bytes — {x[3]}") for x in f.size_outliers],
    ),
    (
        "truncated_imgs",
        lambda f, _: [(x[0], x[1]) for x in f.truncated_imgs],
    ),
    (
        "duplicate_views",
        lambda f, _: [(x[0], f"{x[1]} identical to {x[2]}") for x in f.duplicate_views],
    ),
    (
        "blank_imgs",
        lambda f, _: [(x[0], f"{x[1]}: std={x[2]}") for x in f.blank_imgs],
    ),
    (
        "blurry_imgs",
        lambda f, _: [(x[0], f"{x[1]}: score={x[2]}") for x in f.blurry_imgs],
    ),
    (
        "outside_california",
        lambda f, _: [(x[0], f"{x[1]}: ({x[2]:.6f}, {x[3]:.6f})") for x in f.outside_california],
    ),
    (
        "country_code_issues",
        lambda f, _: [(x[0], f"country={x[1]}") for x in f.country_code_issues],
    ),
    (
        "bad_coords",
        lambda f, _: [(x[0], f"{x[1]}: lat={x[2]}, lon={x[3]}") for x in f.bad_coords],
    ),
    (
        "pano_distance_issues",
        lambda f, _: [(x[0], f"{x[1]:,.0f}m from original") for x in f.pano_distance_issues],
    ),
    (
        "coord_mismatches",
        lambda f, _: [(x[0], f"meta=({x[1]:.6f},{x[2]:.6f}) src=({x[3]:.6f},{x[4]:.6f})") for x in f.coord_mismatches],
    ),
    (
        "missing_meta",
        lambda f, _: [(idx, "no metadata JSON") for idx in sorted(f.missing_meta)],
    ),
    (
        "corrupt_meta",
        lambda f, _: [(x[0], f"JSON error: {x[1]}") for x in f.corrupt_meta],
    ),
    (
        "meta_index_mismatch",
        lambda f, _: [(x[0], f"folder={x[0]:06d} JSON index={x[1]}") for x in f.meta_index_mismatch],
    ),
    (
        "meta_field_issues",
        lambda f, _: [(x[0], f"missing: {x[1]}") for x in f.meta_field_issues],
    ),
    (
        "meta_value_issues",
        lambda f, _: [(x[0], f"{x[1]}: expected {x[2]}, got {x[3]}") for x in f.meta_value_issues],
    ),
    (
        "completed_without_folder",
        lambda _, d: [(idx, "in CSV but no image folder") for idx in sorted(d.completed_without_folder)],
    ),
    (
        "folders_not_in_completed",
        lambda _, d: [(idx, "image folder not in CSV") for idx in sorted(d.folders_not_in_completed)],
    ),
    (
        "orphan_meta",
        lambda _, d: [(idx, "metadata JSON with no image folder") for idx in sorted(d.orphan_meta)],
    ),
    (
        "copyright_issues",
        lambda f, _: [(x[0], f"copyright={repr(x[1])}") for x in f.copyright_issues],
    ),
    (
        "bad_dates",
        lambda f, _: [(x[0], f"date={repr(x[1])}") for x in f.bad_dates],
    ),
    (
        "panoid_csv_mismatches",
        lambda f, _: [(x[0], f"csv={x[1]}  json={x[2]}") for x in f.panoid_csv_mismatches],
    ),
    (
        "duplicate_panoids",
        lambda _, d: [
            (idx, f"panoid={panoid} shared with {len(indices) - 1} other(s)")
            for panoid, indices in d.duplicate_panoids.items()
            for idx in sorted(indices)
        ],
    ),
    (
        "unexpected_files",
        lambda f, _: [(x[0], x[1]) for x in f.unexpected_files],
    ),
    (
        "never_tried",
        lambda _, d: [(idx, "never attempted") for idx in sorted(d.never_tried)],
    ),
    (
        "duplicate_completed",
        lambda _, d: [(idx, f"appears {count}x in completed CSV") for idx, count in sorted(d.duplicate_completed.items())],
    ),
    (
        "duplicate_rejected",
        lambda _, d: [(idx, f"appears {count}x in rejected CSV") for idx, count in sorted(d.duplicate_rejected.items())],
    ),
    (
        "in_both",
        lambda _, d: [(idx, "in both completed and rejected") for idx in sorted(d.in_both)],
    ),
]


def _build_export_sections(findings, derived):
    """Build flagged.txt sections from findings. Returns list of (label, entries) tuples."""
    sections = []
    for label, entries_fn in _EXPORT_SPECS:
        entries = entries_fn(findings, derived)
        if entries:
            sections.append((label, entries))
    return sections


def _print_report(findings, derived, disk_indices, has_metadata_dir, rejected_count, reject_reasons, source_indices):
    """Print the Results section. Returns total issue count."""
    print()
    print("Results")
    print("=" * 60)

    # Dataset summary
    disk_gb = findings.total_disk_bytes / (1024 ** 3)
    print(f"Dataset: {len(disk_indices):,} locations, {findings.total_images_ok:,} valid images, {disk_gb:.1f} GB on disk | {rejected_count:,} rejected")
    print()

    issues = 0

    # [1] Image completeness
    n1_ok  = len(disk_indices) - len(findings.incomplete) - len(findings.empty_folders)
    n1_bad = len(findings.incomplete) + len(findings.empty_folders)
    issues += n1_bad
    tag1     = "PASS" if n1_bad == 0 else "FAIL"
    summary1 = f"{n1_ok:,} OK / {len(findings.incomplete):,} incomplete"
    if findings.empty_folders:
        summary1 += f" / {len(findings.empty_folders):,} empty"
    print(f"[1] Image completeness (4 per folder)    : {tag1} — {summary1}")

    # [2] Image integrity
    n2_bad = (len(findings.corrupt_imgs) + len(findings.bad_dimensions) + len(findings.bad_color_mode)
              + len(findings.blank_imgs) + len(findings.blurry_imgs)
              + len(findings.truncated_imgs) + len(findings.duplicate_views))
    issues  += n2_bad
    tag2     = "PASS" if n2_bad == 0 else "FAIL"
    summary2 = f"{findings.total_images_ok:,} OK / {len(findings.corrupt_imgs):,} corrupt / {len(findings.bad_dimensions):,} wrong size"
    if findings.truncated_imgs:
        summary2 += f" / {len(findings.truncated_imgs):,} truncated"
    if findings.duplicate_views:
        summary2 += f" / {len(findings.duplicate_views):,} duplicate views"
    if findings.bad_color_mode:
        summary2 += f" / {len(findings.bad_color_mode):,} non-RGB"
    if findings.blank_imgs:
        summary2 += f" / {len(findings.blank_imgs):,} blank/degenerate"
    if findings.blurry_imgs:
        summary2 += f" / {len(findings.blurry_imgs):,} blurry"
    print(f"[2] Image integrity (readable, 1024x1024): {tag2} — {summary2}")

    # [3] Metadata integrity
    if has_metadata_dir:
        n3_bad = (len(findings.missing_meta) + len(findings.corrupt_meta)
                  + len(findings.meta_field_issues) + len(findings.meta_value_issues)
                  + len(findings.meta_index_mismatch) + len(findings.coord_mismatches)
                  + len(findings.pano_distance_issues) + len(findings.bad_coords)
                  + len(findings.outside_california) + len(findings.copyright_issues)
                  + len(findings.country_code_issues) + len(findings.bad_dates)
                  + len(findings.panoid_csv_mismatches))
        issues   += n3_bad
        n3_ok     = len(disk_indices) - len(findings.missing_meta) - len(findings.corrupt_meta)
        tag3      = "PASS" if n3_bad == 0 else "FAIL"
        summary3  = f"{n3_ok:,} OK / {len(findings.missing_meta):,} missing / {len(findings.corrupt_meta):,} corrupt"
        if findings.meta_field_issues:
            summary3 += f" / {len(findings.meta_field_issues):,} incomplete fields"
        if findings.meta_value_issues:
            summary3 += f" / {len(findings.meta_value_issues):,} wrong config values"
        if findings.meta_index_mismatch:
            summary3 += f" / {len(findings.meta_index_mismatch):,} index mismatch"
        if findings.coord_mismatches:
            summary3 += f" / {len(findings.coord_mismatches):,} coordinate mismatch"
        if findings.pano_distance_issues:
            summary3 += f" / {len(findings.pano_distance_issues):,} pano too far (>{MAX_PANO_DISTANCE_M}m)"
        if findings.bad_coords:
            summary3 += f" / {len(findings.bad_coords):,} out-of-bounds coords"
        if findings.outside_california:
            summary3 += f" / {len(findings.outside_california):,} outside California"
        if findings.copyright_issues:
            summary3 += f" / {len(findings.copyright_issues):,} non-Google copyright"
        if findings.country_code_issues:
            summary3 += f" / {len(findings.country_code_issues):,} wrong country"
        if findings.bad_dates:
            summary3 += f" / {len(findings.bad_dates):,} bad date format"
        if findings.panoid_csv_mismatches:
            summary3 += f" / {len(findings.panoid_csv_mismatches):,} panoid CSV↔JSON mismatch"
        print(f"[3] Metadata integrity                   : {tag3} — {summary3}")
    else:
        print(f"[3] Metadata integrity                   : SKIP — metadata/ not found")

    # [4] Source CSV coverage
    if source_indices is not None:
        n4_attempted = len(source_indices) - len(derived.never_tried)
        tag4 = "PASS" if len(derived.never_tried) == 0 else "WARN"
        print(f"[4] Source CSV coverage                   : {tag4} — {n4_attempted:,} / {len(source_indices):,} attempted, {len(derived.never_tried):,} never tried")
    else:
        print(f"[4] Source CSV coverage                   : SKIP — no --source-csv provided")

    # [5] Completed ↔ images match
    n5_bad = len(derived.folders_not_in_completed) + len(derived.completed_without_folder)
    issues += n5_bad
    tag5    = "PASS" if n5_bad == 0 else "FAIL"
    print(f"[5] Completed <-> images match            : {tag5} — {len(derived.folders_not_in_completed):,} folders not in completed / {len(derived.completed_without_folder):,} completed without folder")

    # [6] Additional checks
    n6_bad = len(derived.duplicate_completed) + len(derived.in_both) + len(derived.duplicate_panoids)
    issues += n6_bad
    print(f"[6] Additional")
    print(f"    Duplicate completed entries           : {len(derived.duplicate_completed):,}")
    print(f"    Duplicate rejected entries            : {len(derived.duplicate_rejected):,}")
    print(f"    In BOTH completed AND rejected        : {len(derived.in_both):,}")
    print(f"    Duplicate panoid (same pano, diff loc): {len(derived.duplicate_panoids):,}")
    if has_metadata_dir:
        print(f"    Orphan metadata (no image folder)     : {len(derived.orphan_meta):,}")
    if findings.unexpected_files:
        print(f"    Unexpected files in image folders     : {len(findings.unexpected_files):,}")
    if findings.size_outliers:
        print(f"    File size outliers (WARN)             : {len(findings.size_outliers):,}")
    if reject_reasons:
        print(f"    Reject reason breakdown:")
        for reason, count in reject_reasons.most_common():
            print(f"      {reason:40s} {count:,}")

    # Details section
    details = []
    for header, items_fn, formatter in _DETAIL_SPECS:
        details += _detail_block(header, items_fn(findings, derived), formatter)

    if details:
        print("\n" + "-" * 60)
        print("Details")
        print("-" * 60)
        for line in details:
            print(line)

    return issues


# ---------------------------------------------------------------------------
# Worker process shared state — set once per worker via initializer
# ---------------------------------------------------------------------------

_source_coords     = None
_completed_panoids = None


def _init_worker(source_coords, completed_panoids):
    """Initialize worker process with shared dicts (set once, not re-pickled per task)."""
    global _source_coords, _completed_panoids
    _source_coords     = source_coords
    _completed_panoids = completed_panoids


# ---------------------------------------------------------------------------
# Per-folder worker — image and metadata sub-checks
# ---------------------------------------------------------------------------

def _check_images(folder, idx, result):
    """Check all images in a location folder. Mutates result in place."""
    # Single directory listing — split into jpg and non-jpg in one pass
    all_files = list(folder.iterdir())
    jpg_files = sorted(f for f in all_files if f.is_file() and f.suffix.lower() == ".jpg")
    present   = {f.name for f in jpg_files}

    missing = EXPECTED_IMAGES - present
    if not present:
        result["empty"] = True
    elif missing:
        result["incomplete"] = (idx, sorted(missing))

    for f in all_files:
        if f.is_file() and f.suffix.lower() != ".jpg":
            result["unexpected_files"].append((idx, f.name))

    view_hashes = {}
    for img_path in jpg_files:
        try:
            raw = img_path.read_bytes()
        except Exception as e:
            result["corrupt_imgs"].append((idx, img_path.name, f"cannot read: {e}"))
            continue

        file_size = len(raw)
        result["disk_bytes"] += file_size

        if file_size == 0:
            result["corrupt_imgs"].append((idx, img_path.name, "empty file (0 bytes)"))
            continue

        if file_size < MIN_FILE_SIZE:
            result["size_outliers"].append((idx, img_path.name, file_size, "suspiciously small"))
        elif file_size > MAX_FILE_SIZE:
            result["size_outliers"].append((idx, img_path.name, file_size, "suspiciously large"))

        # Truncated JPEG: valid JPEGs end with FFD9
        if raw[-2:] != _JPEG_EOI:
            result["truncated_imgs"].append((idx, img_path.name))

        # Duplicate view detection: hash file content
        file_hash = hashlib.md5(raw, usedforsecurity=False).hexdigest()
        if file_hash in view_hashes:
            result["duplicate_views"].append((idx, img_path.name, view_hashes[file_hash]))
        else:
            view_hashes[file_hash] = img_path.name

        # PIL validation
        try:
            with Image.open(io.BytesIO(raw)) as img:
                img.load()
                w, h = img.size
                if (w, h) != (1024, 1024):
                    result["bad_dimensions"].append((idx, img_path.name, f"{w}x{h}"))
                else:
                    result["images_ok"] += 1
                if img.mode != "RGB":
                    result["bad_color_mode"].append((idx, img_path.name, img.mode))
                # uint8 view — zero-copy buffer from PIL; std on 3MB vs 12MB float32.
                arr_u8 = np.asarray(img)
                std_val = float(arr_u8.std())
                if std_val < BLANK_STD_THRESHOLD:
                    result["blank_imgs"].append((idx, img_path.name, round(std_val, 2)))
                else:
                    # Blur detection: Laplacian variance on grayscale.
                    # Only run for non-blank images; convert to float32 here rather
                    # than upfront so blank images never pay the allocation cost.
                    if arr_u8.ndim == 3 and arr_u8.shape[2] == 3:
                        arr_f = arr_u8.astype(np.float32)
                        gray = _BT601_R * arr_f[:, :, 0] + _BT601_G * arr_f[:, :, 1] + _BT601_B * arr_f[:, :, 2]
                    elif arr_u8.ndim == 2:
                        gray = arr_u8.astype(np.float32)
                    else:
                        gray = np.array(img.convert("L"), dtype=np.float32)  # RGBA/CMYK/etc.
                    laplacian = (
                        gray[:-2, 1:-1] + gray[2:, 1:-1]
                        + gray[1:-1, :-2] + gray[1:-1, 2:]
                        - 4 * gray[1:-1, 1:-1]
                    )
                    blur_score = float(laplacian.var())
                    if blur_score < BLUR_THRESHOLD:
                        result["blurry_imgs"].append((idx, img_path.name, round(blur_score, 2)))
        except Exception as e:
            result["corrupt_imgs"].append((idx, img_path.name, str(e)))


def _check_metadata(meta_path, idx, source_coords, completed_panoids, result):
    """Check one metadata JSON file. Mutates result in place."""
    if not meta_path.exists():
        result["missing_meta"] = True
        return

    try:
        meta_raw = meta_path.read_bytes()
        result["disk_bytes"] += len(meta_raw)
        meta = json.loads(meta_raw)

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

        # Copyright check — must contain "Google" to confirm official panorama
        copyright_val = meta.get("copyright", "")
        if not copyright_val or "Google" not in str(copyright_val):
            result["copyright_issues"].append((idx, str(copyright_val)))

        # Country code — all California data must be US
        country_code = meta.get("country_code", "")
        if country_code and country_code != "US":
            result["country_code_issues"].append((idx, country_code))

        # Capture date format — must be YYYY-MM if present
        date_val = meta.get("date", "")
        if date_val and not _DATE_RE.match(str(date_val)):
            result["bad_dates"].append((idx, str(date_val)))

        # Panoid cross-check: completed CSV recorded panoid vs metadata panoid
        if completed_panoids and idx in completed_panoids:
            csv_panoid  = completed_panoids[idx]
            meta_panoid = meta.get("panoid", "")
            if csv_panoid and meta_panoid and csv_panoid != meta_panoid:
                result["panoid_csv_mismatches"].append((idx, csv_panoid, meta_panoid))

        if source_coords and idx in source_coords:
            src_lat, src_lon = source_coords[idx]
            meta_lat = meta.get("original_lat")
            meta_lon = meta.get("original_lon")
            if meta_lat is not None and meta_lon is not None:
                if (abs(float(meta_lat) - src_lat) > COORD_TOLERANCE
                        or abs(float(meta_lon) - src_lon) > COORD_TOLERANCE):
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

        for field_name, lat_key, lon_key in [
            ("original", "original_lat", "original_lon"),
            ("pano",     "pano_lat",     "pano_lon"),
        ]:
            lat_v = meta.get(lat_key)
            lon_v = meta.get(lon_key)
            if lat_v is not None and lon_v is not None:
                try:
                    la, lo = float(lat_v), float(lon_v)
                    if not (-90 <= la <= 90) or not (-180 <= lo <= 180):
                        result["bad_coords"].append((idx, field_name, la, lo))
                    elif not (CA_LAT_MIN <= la <= CA_LAT_MAX
                              and CA_LON_MIN <= lo <= CA_LON_MAX):
                        result["outside_california"].append((idx, field_name, la, lo))
                except (ValueError, TypeError):
                    result["bad_coords"].append((idx, field_name, lat_v, lon_v))

    except Exception as e:
        result["corrupt_meta"] = (idx, str(e))


def _process_folder(folder, has_metadata_dir, metadata_dir):
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
        blurry_imgs=[],
        truncated_imgs=[],
        duplicate_views=[],
        unexpected_files=[],
        images_ok=0,
        disk_bytes=0,
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
        copyright_issues=[],
        country_code_issues=[],
        bad_dates=[],
        panoid_csv_mismatches=[],
    )

    try:
        _check_images(folder, idx, result)
    except Exception as e:
        result["corrupt_imgs"].append((idx, "?", f"unreadable folder: {e}"))

    if has_metadata_dir:
        meta_path = metadata_dir / f"{idx:06d}.json"
        _check_metadata(meta_path, idx, _source_coords, _completed_panoids, result)

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t0 = time.time()
    args = _parse_args()

    salty_data   = Path(args.data_dir)
    images_dir   = salty_data / "images"
    metadata_dir = salty_data / "metadata"

    print("SALTY Integrity Check")
    print("=" * 60)
    print(f"Directory: {salty_data.resolve()}")

    has_metadata_dir = _validate_dirs(salty_data, images_dir)

    print()
    print("Loading CSVs...")
    csv_data = _load_csvs(salty_data)

    source_coords, source_indices = None, None
    if args.source_csv:
        source_coords, source_indices = _load_source_csv(Path(args.source_csv))

    folders, disk_indices = _collect_folders(images_dir)
    print(f"\nScanning {len(folders):,} image folders with {args.workers} workers...")

    findings = _scan_all_folders(
        folders, has_metadata_dir, metadata_dir,
        source_coords, csv_data["completed_panoids"], args.workers,
    )
    derived = _compute_derived(
        findings, csv_data, disk_indices, has_metadata_dir, metadata_dir, source_indices,
    )

    issues = _print_report(
        findings, derived, disk_indices, has_metadata_dir,
        len(csv_data["rejected_set"]), csv_data["reject_reasons"], source_indices,
    )

    elapsed = time.time() - t0
    print()
    print("=" * 60)
    print("All checks passed." if issues == 0 else f"{issues:,} issue(s) found.")
    print(f"Completed in {elapsed:.1f}s")

    if not args.no_export:
        sections     = _build_export_sections(findings, derived)
        flagged_path = salty_data / "flagged.txt"
        n_flagged    = write_flagged_export(flagged_path, sections)
        if n_flagged > 0:
            print(f"Flagged export: {flagged_path.name} ({n_flagged:,} entries)")
        else:
            print("Flagged export: nothing to flag")

    print()
    sys.exit(0 if issues == 0 else 1)


if __name__ == "__main__":
    main()
