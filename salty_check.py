"""
SALTY Data Integrity Checker (read-only)
Validates completeness and health of a salty_data/ download.

Usage:
    uv run salty_check.py salty_data
    uv run salty_check.py salty_data.tar
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
import tarfile
import time
from array import array
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal, InvalidOperation
from itertools import batched
from math import asin, cos, radians, sin, sqrt
from pathlib import Path, PurePosixPath

from PIL import Image, ImageStat
import numpy as np
import pandas as pd
import simplejpeg
from tqdm import tqdm

_DATE_RE = re.compile(r"^\d{4}-\d{2}$")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXPECTED_IMAGE_NAMES = ("000.jpg", "090.jpg", "180.jpg", "270.jpg")
EXPECTED_IMAGES = set(EXPECTED_IMAGE_NAMES)
_EXPECTED_IMAGE_SLOT = {name: slot for slot, name in enumerate(EXPECTED_IMAGE_NAMES)}
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

# Locations sent to a worker per process-pool task. Batching cuts IPC
# overhead while keeping only a small amount of work queued at once.
SCAN_BATCH_SIZE = 16

# Keep the throughput display readable without redrawing it for every fast batch.
SCAN_PROGRESS_MIN_INTERVAL_S = 1.0

# ITU-R BT.601 luminance weights for fast RGB -> grayscale (avoids second PIL decode)
_BT601_R, _BT601_G, _BT601_B = 0.299, 0.587, 0.114

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
    regional_blurry_imgs:  list = field(default_factory=list)   # list[tuple[int, str, str]] (side)
    truncated_imgs:        list = field(default_factory=list)   # list[tuple[int, str]]
    duplicate_views:       list = field(default_factory=list)   # list[tuple[int, str, str]]
    unexpected_files:      list = field(default_factory=list)   # list[tuple[int, str]]
    missing_meta:          list = field(default_factory=list)   # list[int]
    corrupt_meta:          list = field(default_factory=list)   # list[tuple[int, str]]
    meta_field_issues:     list = field(default_factory=list)   # list[tuple[int, list[str]]]
    meta_value_issues:     list = field(default_factory=list)   # list[tuple[int, str, Any, Any]]
    meta_index_mismatch:   list = field(default_factory=list)   # list[tuple[int, Any]]
    coord_mismatches:      list = field(default_factory=list)   # list[tuple[int, float, float, float, float]]
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


@dataclass(frozen=True, slots=True)
class ArchiveMember:
    """Location of one regular file inside an uncompressed tar archive."""
    name: str
    offset: int
    size: int

    @property
    def suffix(self):
        return PurePosixPath(self.name).suffix

    def exists(self):
        return True

    def read_bytes(self):
        return _archive_member_bytes(self)


@dataclass(frozen=True, slots=True)
class ArchiveFolder:
    """All archive members needed to check one image folder."""
    idx: int
    images: tuple
    metadata: object = None


@dataclass(slots=True)
class ArchiveFolderIndex:
    """Compact, lazily materialized index of archive location folders."""
    indices: array
    image_offsets: array
    image_sizes: array
    metadata_offsets: array
    metadata_sizes: array
    extras: dict
    order: array

    def __len__(self):
        return len(self.order)

    def __iter__(self):
        for pos in self.order:
            idx = self.indices[pos]
            images = []
            base = pos * len(EXPECTED_IMAGE_NAMES)
            for slot, name in enumerate(EXPECTED_IMAGE_NAMES):
                offset = self.image_offsets[base + slot]
                if offset >= 0:
                    images.append(ArchiveMember(name, offset, self.image_sizes[base + slot]))
            images.extend(self.extras.get(pos, ()))
            images.sort(key=lambda member: member.name)

            metadata = None
            if self.metadata_offsets[pos] >= 0:
                metadata = ArchiveMember(
                    f"{idx:06d}.json",
                    self.metadata_offsets[pos],
                    self.metadata_sizes[pos],
                )
            yield ArchiveFolder(idx, tuple(images), metadata)


@dataclass
class ArchiveDataset:
    """Indexed view of a salty_data tree stored in an uncompressed tar."""
    folders: ArchiveFolderIndex
    disk_indices: set
    metadata_indices: set
    has_metadata_dir: bool
    completed_files: list
    rejected_files: list
    root_name: str


class NamedBytesIO(io.BytesIO):
    """BytesIO carrying a filename for pandas loaders and warning messages."""

    def __init__(self, raw, name):
        super().__init__(raw)
        self.name = name


class InputValidationError(ValueError):
    """A required checker input cannot be used safely."""


@dataclass
class InputDiagnostics:
    """Warnings found while validating inputs before the image scan."""
    warning_count: int = 0

    def warn(self, message, count=1):
        print(f"  WARNING: {message}")
        self.warning_count += count


class _ArchiveIndexBuilder:
    """Mutable compact index for one possible dataset root."""

    def __init__(self, root):
        self.root = root
        self.score = 0
        self.first_score_order = None
        self.positions = {}
        self.indices = array("q")
        self.image_present = bytearray()
        self.image_file_present = bytearray()
        self.first_offsets = array("q")
        self.image_offsets = array("q")
        self.image_sizes = array("q")
        self.metadata_offsets = array("q")
        self.metadata_sizes = array("q")
        self.extras = {}
        self.metadata_indices = set()
        self.has_metadata_dir = False
        self.completed_members = []
        self.rejected_members = []

    def _position(self, idx):
        pos = self.positions.get(idx)
        if pos is not None:
            return pos
        pos = len(self.indices)
        self.positions[idx] = pos
        self.indices.append(idx)
        self.image_present.append(False)
        self.image_file_present.append(False)
        self.first_offsets.append(-1)
        self.image_offsets.extend([-1] * len(EXPECTED_IMAGE_NAMES))
        self.image_sizes.extend([0] * len(EXPECTED_IMAGE_NAMES))
        self.metadata_offsets.append(-1)
        self.metadata_sizes.append(0)
        return pos

    def _mark_image_folder(self, idx, offset, is_file=False):
        pos = self._position(idx)
        self.image_present[pos] = True
        current = self.first_offsets[pos]
        if is_file and (not self.image_file_present[pos] or offset < current):
            self.image_file_present[pos] = True
            self.first_offsets[pos] = offset
        elif not self.image_file_present[pos] and (current < 0 or offset < current):
            self.first_offsets[pos] = offset
        return pos

    def add_image_directory(self, idx, offset):
        self._mark_image_folder(idx, offset)

    def add_image_file(self, idx, name, offset, size):
        pos = self._mark_image_folder(idx, offset, is_file=True)
        slot = _EXPECTED_IMAGE_SLOT.get(name)
        if slot is None:
            self.extras.setdefault(pos, []).append(ArchiveMember(name, offset, size))
            return
        array_pos = pos * len(EXPECTED_IMAGE_NAMES) + slot
        if self.image_offsets[array_pos] < 0:
            self.image_offsets[array_pos] = offset
            self.image_sizes[array_pos] = size
        else:
            # Preserve duplicate tar members so the checker sees exactly the
            # same member list as the previous archive implementation.
            self.extras.setdefault(pos, []).append(ArchiveMember(name, offset, size))

    def add_metadata_file(self, filename, offset, size):
        self.has_metadata_dir = True
        path = PurePosixPath(filename)
        if path.suffix != ".json":
            return
        try:
            idx = int(path.stem)
        except ValueError:
            return
        self.metadata_indices.add(idx)
        if filename != f"{idx:06d}.json":
            return
        pos = self._position(idx)
        # Like tar extraction and the previous dict index, the final metadata
        # member with a duplicate pathname wins.
        self.metadata_offsets[pos] = offset
        self.metadata_sizes[pos] = size

    def build_folder_index(self):
        positions = [
            pos for pos, present in enumerate(self.image_present) if present
        ]
        positions.sort(key=lambda pos: (self.first_offsets[pos], self.indices[pos]))
        extras = {pos: tuple(members) for pos, members in self.extras.items()}
        return ArchiveFolderIndex(
            indices=self.indices,
            image_offsets=self.image_offsets,
            image_sizes=self.image_sizes,
            metadata_offsets=self.metadata_offsets,
            metadata_sizes=self.metadata_sizes,
            extras=extras,
            order=array("q", positions),
        )


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


def _parse_index(value):
    """Parse an integer index without rounding or truncating invalid values."""
    if isinstance(value, bool):
        raise ValueError("boolean index")
    try:
        number = Decimal(str(value))
    except InvalidOperation as exc:
        raise ValueError("invalid index") from exc
    if not number.is_finite() or number != number.to_integral_value():
        raise ValueError("non-integer index")
    return int(number)


def _csv_index_values(series, csv_file, diagnostics=None, report_warning=True):
    """Preserve valid rows while reporting invalid indices consistently."""
    values = []
    for value in series:
        try:
            values.append(_parse_index(value))
        except ValueError:
            values.append(None)
    bad_count = sum(value is None for value in values)
    if bad_count and report_warning:
        message = (
            f"{bad_count} non-integer value(s) in index column of "
            f"{csv_file.name} - skipping those rows"
        )
        if diagnostics is None:
            print(f"  WARNING: {message}")
        else:
            diagnostics.warn(message, bad_count)
    return pd.Series(values, index=series.index, dtype=object)


def load_csv_indices(files, diagnostics=None):
    """Load index values from one or more CSVs. Returns (unique set, raw list)."""
    raw = []
    for csv_file in files:
        try:
            if hasattr(csv_file, "seek"):
                csv_file.seek(0)
            series = pd.read_csv(csv_file, dtype={"index": str})["index"]
            indices = _csv_index_values(series, csv_file, diagnostics)
            raw.extend(indices.dropna().tolist())
        except Exception as e:
            raise InputValidationError(f"could not read {csv_file.name}: {e}") from e
    return set(raw), raw


def load_completed_panoids(files, report_invalid=True):
    """Load {index: panoid} from completed CSVs. Used for CSV↔metadata cross-check."""
    result = {}
    for csv_file in files:
        try:
            if hasattr(csv_file, "seek"):
                csv_file.seek(0)
            df = pd.read_csv(csv_file, dtype={"index": str})
            if "panoid" in df.columns and "index" in df.columns:
                sub = df[["index", "panoid"]].dropna(subset=["panoid"])
                sub = sub.copy()
                sub["index"] = _csv_index_values(
                    sub["index"], csv_file, report_warning=report_invalid,
                )
                sub = sub.dropna(subset=["index"]).astype({"panoid": str})
                result.update(zip(sub["index"], sub["panoid"]))
        except Exception as e:
            raise InputValidationError(
                f"could not read panoids from {csv_file.name}: {e}"
            ) from e
    return result


def load_source_coords(source_path, diagnostics=None):
    """Load source CSV and return {index: (lat, lon)} dict."""
    df = pd.read_csv(source_path, dtype=str)
    if df.shape[1] < 3:
        raise InputValidationError(
            f"{source_path.name} must contain index, latitude, and longitude columns"
        )
    idx_col = _csv_index_values(df.iloc[:, 0], source_path, diagnostics)
    valid = idx_col.notna()
    df = df.loc[valid]
    idx_col = idx_col.loc[valid]
    if df.empty:
        raise InputValidationError(f"{source_path.name} contains no usable index rows")
    try:
        lat_col = df.iloc[:, 1].astype(float)
        lon_col = df.iloc[:, 2].astype(float)
    except (TypeError, ValueError) as exc:
        raise InputValidationError(
            f"{source_path.name} contains a non-numeric latitude or longitude"
        ) from exc
    finite = np.isfinite(lat_col.to_numpy()) & np.isfinite(lon_col.to_numpy())
    if not finite.all():
        raise InputValidationError(
            f"{source_path.name} contains a non-finite latitude or longitude"
        )
    return dict(zip(idx_col, zip(lat_col, lon_col)))


def load_reject_reasons(files):
    """Load rejection reasons from rejects CSVs. Returns Counter of reasons."""
    reasons = Counter()
    for csv_file in files:
        try:
            if hasattr(csv_file, "seek"):
                csv_file.seek(0)
            df = pd.read_csv(csv_file)
            if "reason" in df.columns:
                reasons.update(df["reason"].dropna().values.tolist())
        except Exception as e:
            raise InputValidationError(
                f"could not read rejection reasons from {csv_file.name}: {e}"
            ) from e
    return reasons


def write_flagged_export(path, sections):
    """Write flagged indices to a review file. Returns count of entries written."""
    total = 0
    lines = [
        f"# SALTY flagged locations — {date.today()}",
        "# Entries are commented out by default (safe). Remove the leading '# ' to reject.",
        "# Run: uv run salty_reject.py <data_dir> --from-file flagged.txt",
    ]
    for label, entries in sections:
        if not entries:
            continue
        lines.append("#")
        lines.append(f"# --- {label} ({len(entries):,}) ---")
        for idx, detail in entries:
            lines.append(f"# {idx:06d}  # {label}: {detail}")
            total += 1
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

def _positive_int(value):
    """Parse a strictly positive command-line integer."""
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def _parse_args():
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="SALTY data integrity checker")
    parser.add_argument(
        "data_dir",
        help="Path to a salty_data directory or uncompressed .tar archive",
    )
    parser.add_argument(
        "--source-csv",
        default=None,
        help="Path to source coordinate CSV (e.g. 100k-205k_data.csv) for coverage check",
    )
    parser.add_argument(
        "--workers",
        type=_positive_int,
        default=os.cpu_count() or 1,
        help=f"Number of parallel scan workers (default: {os.cpu_count() or 1})",
    )
    parser.add_argument(
        "--no-export",
        action="store_true",
        help="Do not write flagged.txt (default: always write)",
    )
    return parser.parse_args()


def _validate_dirs(salty_data, images_dir, diagnostics=None):
    """Validate required directories exist. Returns has_metadata_dir flag."""
    if not salty_data.is_dir():
        raise InputValidationError(f"{salty_data} is not a directory")
    if not images_dir.exists():
        raise InputValidationError(f"{images_dir} not found")
    if not images_dir.is_dir():
        raise InputValidationError(f"{images_dir} is not a directory")
    metadata_dir = salty_data / "metadata"
    has_metadata_dir = metadata_dir.exists()
    if has_metadata_dir and not metadata_dir.is_dir():
        raise InputValidationError(f"{metadata_dir} is not a directory")
    if not has_metadata_dir:
        message = f"{metadata_dir} not found — metadata checks will be skipped"
        if diagnostics is None:
            print(f"  WARNING: {message}")
        else:
            diagnostics.warn(message)
    return has_metadata_dir


def _load_csvs(salty_data, diagnostics=None):
    """Discover and load completed + rejected CSVs. Returns dict of all derived data."""
    completed_files = sorted(salty_data.glob("completed*.csv"))
    rejected_files  = sorted(salty_data.glob("rejects*.csv"))

    return _load_csv_file_lists(completed_files, rejected_files, diagnostics)


def _load_csv_file_lists(completed_files, rejected_files, diagnostics=None):
    """Load already-discovered completed and rejected CSV file objects."""

    completed_set, completed_raw = set(), []
    completed_panoids = {}
    if completed_files:
        completed_set, completed_raw = load_csv_indices(completed_files, diagnostics)
        # The index loader already reported malformed rows from these files.
        completed_panoids = load_completed_panoids(completed_files, report_invalid=False)
        names = ", ".join(csv_file.name for csv_file in completed_files)
        print(f"  completed : {names}")
        print(f"            -> {len(completed_set):,} unique entries")
    else:
        message = "No completed*.csv found"
        if diagnostics is None:
            print(f"  WARNING: {message}")
        else:
            diagnostics.warn(message)

    rejected_set, rejected_raw = set(), []
    reject_reasons = Counter()
    if rejected_files:
        rejected_set, rejected_raw = load_csv_indices(rejected_files, diagnostics)
        reject_reasons = load_reject_reasons(rejected_files)
        names = ", ".join(csv_file.name for csv_file in rejected_files)
        print(f"  rejects   : {names}")
        print(f"            -> {len(rejected_set):,} unique entries")
    else:
        message = "No rejects*.csv found"
        if diagnostics is None:
            print(f"  WARNING: {message}")
        else:
            diagnostics.warn(message)

    return dict(
        completed_set=completed_set,
        completed_raw=completed_raw,
        completed_panoids=completed_panoids,
        rejected_set=rejected_set,
        rejected_raw=rejected_raw,
        reject_reasons=reject_reasons,
    )


def _tar_parts(name):
    """Return safe, normalized POSIX member-name components."""
    raw_parts = name.split("/")
    if raw_parts and raw_parts[0] and all(
        part not in ("", ".", "..") for part in raw_parts
    ):
        return tuple(raw_parts)

    # Preserve PurePosixPath normalization for unusual names while keeping
    # the common archive path free of per-member Path object allocation.
    parts = tuple(part for part in PurePosixPath(name).parts if part not in (".", "/"))
    if not parts or ".." in parts:
        return None
    return parts


def _read_archive_member(archive_path, member):
    """Read one member directly by its data offset in an uncompressed tar."""
    with archive_path.open("rb") as archive_file:
        return _read_archive_bytes(archive_file, member)


def _read_archive_bytes(archive_file, member):
    """Read an indexed member from an open archive, rejecting short reads."""
    archive_file.seek(member.offset)
    raw = archive_file.read(member.size)
    if len(raw) != member.size:
        raise OSError(f"short read: expected {member.size:,} bytes, got {len(raw):,}")
    return raw


def _archive_image_location(info, parts):
    """Return (dataset root, relative path, index) for an image folder member."""
    if info.isfile():
        for pos, part in enumerate(parts[:-1]):
            if part != "images":
                continue
            # Only direct children of images/<numeric-folder>/ belong to a
            # location. Deeper descendants must not affect root selection.
            if pos + 3 != len(parts):
                continue
            try:
                idx = int(parts[pos + 1])
            except ValueError:
                continue
            return parts[:pos], parts[pos:], idx
    elif info.isdir() and len(parts) >= 2 and parts[-2] == "images":
        try:
            idx = int(parts[-1])
        except ValueError:
            pass
        else:
            return parts[:-2], parts[-2:], idx
    return None


def _iter_tar_members(archive_path):
    """Stream tar headers with progress, without retaining them in TarFile."""
    archive_size = archive_path.stat().st_size
    progress_offset = 0
    header_count = 0
    try:
        with tqdm(
            total=archive_size,
            desc="Indexing archive",
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            smoothing=0.3,
        ) as progress:
            with tarfile.open(archive_path, mode="r:") as archive:
                while True:
                    info = archive.next()
                    if info is None:
                        break
                    # TarFile normally retains every TarInfo. Offsets are all
                    # this checker needs, so release headers as they are read.
                    archive.members.clear()
                    header_count += 1
                    if header_count % 1024 == 0:
                        current = min(archive.offset, archive_size)
                        progress.update(current - progress_offset)
                        progress_offset = current
                    yield info

            progress.update(archive_size - progress_offset)
    except (tarfile.TarError, OSError) as exc:
        raise ValueError(f"could not read uncompressed tar archive: {exc}") from exc


def _index_tar_archive(archive_path):
    """Index a salty_data tree in an uncompressed tar without extracting it."""
    builders = {}
    scored_roots = 0

    def builder_for(root):
        builder = builders.get(root)
        if builder is None:
            builder = _ArchiveIndexBuilder(root)
            builders[root] = builder
        return builder

    for info in _iter_tar_members(archive_path):
        parts = _tar_parts(info.name)
        if parts is None:
            continue

        image_match = _archive_image_location(info, parts)
        if image_match is not None:
            root, relative, idx = image_match
            candidate = builder_for(root)
            if candidate.score == 0:
                candidate.first_score_order = scored_roots
                scored_roots += 1
            candidate.score += 1
            if info.isdir():
                candidate.add_image_directory(idx, info.offset)
            elif len(relative) == 3:
                candidate.add_image_file(idx, relative[2], info.offset_data, info.size)

        if info.isdir() and parts[-1] == "metadata":
            builder_for(parts[:-1]).has_metadata_dir = True
        elif info.isfile():
            if len(parts) >= 2 and parts[-2] == "metadata":
                builder_for(parts[:-2]).add_metadata_file(
                    parts[-1], info.offset_data, info.size,
                )
            elif parts[-1].startswith("completed") and parts[-1].endswith(".csv"):
                builder_for(parts[:-1]).completed_members.append(
                    ArchiveMember(parts[-1], info.offset_data, info.size),
                )
            elif parts[-1].startswith("rejects") and parts[-1].endswith(".csv"):
                builder_for(parts[:-1]).rejected_members.append(
                    ArchiveMember(parts[-1], info.offset_data, info.size),
                )

    candidates = [builder for builder in builders.values() if builder.score]
    if not candidates:
        raise ValueError("archive contains no images/<numeric-folder>/ dataset")
    selected = max(
        candidates,
        key=lambda candidate: (
            candidate.score, -len(candidate.root), -candidate.first_score_order,
        ),
    )
    folders = selected.build_folder_index()
    disk_indices = {
        folders.indices[pos] for pos in folders.order
    }

    def csv_streams(members):
        return [
            NamedBytesIO(_read_archive_member(archive_path, member), member.name)
            for member in sorted(members, key=lambda item: item.name)
        ]

    return ArchiveDataset(
        folders=folders,
        disk_indices=disk_indices,
        metadata_indices=selected.metadata_indices,
        has_metadata_dir=selected.has_metadata_dir,
        completed_files=csv_streams(selected.completed_members),
        rejected_files=csv_streams(selected.rejected_members),
        root_name="/".join(selected.root) or ".",
    )


def _load_source_csv(source_path, diagnostics=None):
    """Load and validate an explicitly requested source coordinate CSV."""
    if not source_path.exists():
        raise InputValidationError(f"--source-csv {source_path} not found")
    if not source_path.is_file():
        raise InputValidationError(f"--source-csv {source_path} is not a file")
    try:
        source_coords = load_source_coords(source_path, diagnostics)
    except InputValidationError:
        raise
    except Exception as e:
        raise InputValidationError(
            f"could not load --source-csv {source_path.name}: {e}"
        ) from e
    source_indices = set(source_coords.keys())
    print(f"  source    : {source_path.name} -> {len(source_indices):,} entries")
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


def _merge_folder_result(findings, folder_result):
    """Merge one worker result into the scan-wide findings."""
    if folder_result is None:
        return
    if folder_result["empty"]:
        findings.empty_folders.append(folder_result["idx"])
    if folder_result["incomplete"]:
        findings.incomplete.append(folder_result["incomplete"])
    if folder_result["missing_meta"]:
        findings.missing_meta.append(folder_result["idx"])
    for key, target in [
        ("corrupt_meta", findings.corrupt_meta),
        ("meta_field_issues", findings.meta_field_issues),
        ("meta_index_mismatch", findings.meta_index_mismatch),
        ("coord_mismatch", findings.coord_mismatches),
        ("pano_distance_issue", findings.pano_distance_issues),
    ]:
        if folder_result[key]:
            target.append(folder_result[key])
    if folder_result["panoid"]:
        findings.panoid_map.setdefault(folder_result["panoid"], []).append(folder_result["idx"])

    findings.total_images_ok += folder_result["images_ok"]
    findings.total_disk_bytes += folder_result["disk_bytes"]
    for key, target in [
        ("corrupt_imgs", findings.corrupt_imgs),
        ("bad_dimensions", findings.bad_dimensions),
        ("bad_color_mode", findings.bad_color_mode),
        ("size_outliers", findings.size_outliers),
        ("blank_imgs", findings.blank_imgs),
        ("blurry_imgs", findings.blurry_imgs),
        ("regional_blurry_imgs", findings.regional_blurry_imgs),
        ("truncated_imgs", findings.truncated_imgs),
        ("duplicate_views", findings.duplicate_views),
        ("unexpected_files", findings.unexpected_files),
        ("meta_value_issues", findings.meta_value_issues),
        ("bad_coords", findings.bad_coords),
        ("outside_california", findings.outside_california),
        ("copyright_issues", findings.copyright_issues),
        ("country_code_issues", findings.country_code_issues),
        ("bad_dates", findings.bad_dates),
        ("panoid_csv_mismatches", findings.panoid_csv_mismatches),
    ]:
        target.extend(folder_result[key])


def _batch_lookups(batch, source_coords, completed_panoids):
    """Return only the CSV lookup entries needed by one folder batch."""
    indices = []
    for folder in batch:
        try:
            indices.append(folder.idx)
        except AttributeError:
            try:
                indices.append(int(folder.name))
            except ValueError:
                pass
    source_batch = None
    if source_coords:
        source_batch = {idx: source_coords[idx] for idx in indices if idx in source_coords}
    panoid_batch = {}
    if completed_panoids:
        panoid_batch = {
            idx: completed_panoids[idx] for idx in indices if idx in completed_panoids
        }
    return source_batch, panoid_batch


def _collect_batched_scan(folders, submit_batch, n_workers):
    """Collect a bounded set of process-pool batches into ScanFindings."""
    findings = ScanFindings()
    batch_iter = iter(batched(folders, SCAN_BATCH_SIZE))
    max_pending = max(1, n_workers * 2)
    pending = set()

    for batch in batch_iter:
        pending.add(submit_batch(batch))
        if len(pending) >= max_pending:
            break

    with tqdm(
        total=len(folders), desc="Scanning", unit="loc", smoothing=0.3,
        mininterval=SCAN_PROGRESS_MIN_INTERVAL_S,
    ) as progress:
        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                batch_results = future.result()
                for folder_result in batch_results:
                    _merge_folder_result(findings, folder_result)
                progress.update(len(batch_results))
                try:
                    batch = next(batch_iter)
                except StopIteration:
                    continue
                pending.add(submit_batch(batch))
    return findings


def _scan_all_folders(folders, has_metadata_dir, metadata_dir, source_coords, completed_panoids, n_workers):
    """Run a bounded, batched parallel filesystem scan."""

    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_init_worker,
        initargs=(None, None),
    ) as executor:
        def submit_batch(batch):
            source_batch, panoid_batch = _batch_lookups(
                batch, source_coords, completed_panoids,
            )
            return executor.submit(
                _process_folder_batch,
                batch, has_metadata_dir, metadata_dir, source_batch, panoid_batch,
            )

        return _collect_batched_scan(folders, submit_batch, n_workers)


def _scan_archive_folders(archive_path, folders, has_metadata_dir, source_coords, completed_panoids, n_workers):
    """Run a bounded, batched scan against an uncompressed tar."""
    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_init_worker,
        initargs=(None, None, archive_path),
    ) as executor:
        def submit_batch(batch):
            source_batch, panoid_batch = _batch_lookups(
                batch, source_coords, completed_panoids,
            )
            return executor.submit(
                _process_archive_batch,
                batch, has_metadata_dir, source_batch, panoid_batch,
            )

        return _collect_batched_scan(folders, submit_batch, n_workers)


def _compute_derived(findings, csv_data, disk_indices, has_metadata_dir, metadata_source, source_indices):
    """Compute post-scan derived checks. Returns DerivedFindings."""
    derived = DerivedFindings()

    if source_indices is not None:
        attempted = csv_data["completed_set"] | csv_data["rejected_set"]
        derived.never_tried = source_indices - attempted

    derived.folders_not_in_completed = disk_indices - csv_data["completed_set"] - csv_data["rejected_set"]
    derived.completed_without_folder = csv_data["completed_set"] - disk_indices

    completed_counts = Counter(csv_data["completed_raw"])
    derived.duplicate_completed = {idx: n for idx, n in completed_counts.items() if n > 1}

    rejected_counts = Counter(csv_data["rejected_raw"])
    derived.duplicate_rejected = {idx: n for idx, n in rejected_counts.items() if n > 1}

    derived.in_both = csv_data["completed_set"] & csv_data["rejected_set"]

    if has_metadata_dir:
        if isinstance(metadata_source, set):
            derived.orphan_meta = metadata_source - disk_indices
        else:
            for meta_file in metadata_source.iterdir():
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
        "Truncated JPEGs — incomplete image data",
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
        "Large regional blur",
        lambda f, _: f.regional_blurry_imgs,
        lambda x: f"{x[0]:06d}/{x[1]}  blurred {x[2]} side beside sharp scenery",
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
        lambda x: f"{x[0]}  -> indices {sorted(x[1])[:5]}{'...' if len(x[1]) > 5 else ''}",
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
        "Unaccounted image folders (not in completed or rejected)",
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
# Each entry: (label, entries_fn(findings, derived) -> list[tuple[int, str]])
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
        "regional_blurry_imgs",
        lambda f, _: [(x[0], f"{x[1]}: blurred {x[2]} side") for x in f.regional_blurry_imgs],
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
        lambda _, d: [(idx, "unaccounted image folder (not in completed or rejected)") for idx in sorted(d.folders_not_in_completed)],
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


def _print_report(
    findings, derived, disk_indices, has_metadata_dir, rejected_count,
    reject_reasons, source_indices, input_warnings=0,
):
    """Print the Results section. Returns (failure count, warning count)."""
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
              + len(findings.blank_imgs) + len(findings.blurry_imgs) + len(findings.regional_blurry_imgs)
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
    if findings.regional_blurry_imgs:
        summary2 += f" / {len(findings.regional_blurry_imgs):,} regional blur"
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
    print(f"[5] Completed <-> images match            : {tag5} — {len(derived.folders_not_in_completed):,} unaccounted folders / {len(derived.completed_without_folder):,} completed without folder")

    # [6] Additional checks
    n6_bad = len(derived.duplicate_completed) + len(derived.in_both) + len(derived.duplicate_panoids)
    issues += n6_bad
    print(f"[6] Additional")
    print(f"    Duplicate completed entries           : {len(derived.duplicate_completed):,}")
    print(f"    Duplicate rejected entries (WARN)     : {len(derived.duplicate_rejected):,}")
    print(f"    In BOTH completed AND rejected        : {len(derived.in_both):,}")
    print(f"    Duplicate panoid (same pano, diff loc): {len(derived.duplicate_panoids):,}")
    if has_metadata_dir:
        print(f"    Orphan metadata (WARN, no folder)     : {len(derived.orphan_meta):,}")
    if findings.unexpected_files:
        print(f"    Unexpected files in folders (WARN)    : {len(findings.unexpected_files):,}")
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

    warnings = (input_warnings + len(derived.duplicate_rejected) + len(findings.unexpected_files)
                + len(findings.size_outliers))
    if has_metadata_dir:
        warnings += len(derived.orphan_meta)
    if source_indices is not None:
        warnings += len(derived.never_tried)
    return issues, warnings


# ---------------------------------------------------------------------------
# Worker process shared state — set once per worker via initializer
# ---------------------------------------------------------------------------

_source_coords     = None
_completed_panoids = None
_archive_file      = None
_UNSET              = object()


def _set_worker_lookups(source_coords, completed_panoids):
    """Set the lookup subset used by the current worker task."""
    global _source_coords, _completed_panoids
    _source_coords = source_coords
    _completed_panoids = completed_panoids


def _init_worker(source_coords, completed_panoids, archive_path=None):
    """Initialize worker lookups and the optional process-local archive handle."""
    global _archive_file
    _set_worker_lookups(source_coords, completed_panoids)
    if _archive_file is not None:
        _archive_file.close()
    _archive_file = archive_path.open("rb") if archive_path is not None else None


# ---------------------------------------------------------------------------
# Per-folder worker — image and metadata sub-checks
# ---------------------------------------------------------------------------

def _check_image_members(members, idx, result, read_bytes):
    """Check folder completeness, then validate each JPEG member."""
    jpg_files = []
    for f in members:
        if f.suffix.lower() == ".jpg":
            jpg_files.append(f)
        if f.name not in EXPECTED_IMAGES:
            result["unexpected_files"].append((idx, f.name))
    jpg_files.sort(key=lambda p: p.name)
    present = {f.name for f in jpg_files}

    missing = EXPECTED_IMAGES - present
    if not present:
        result["empty"] = True
    elif missing:
        result["incomplete"] = (idx, sorted(missing))

    view_hashes = {}
    for img_path in jpg_files:
        _check_image(img_path, idx, result, read_bytes, view_hashes)


def _check_image(img_path, idx, result, read_bytes, view_hashes):
    """Read and decode one image, recording file and content problems."""
    try:
        raw = read_bytes(img_path)
    except Exception as e:
        result["corrupt_imgs"].append((idx, img_path.name, f"cannot read: {e}"))
        return

    file_size = len(raw)
    result["disk_bytes"] += file_size

    if file_size == 0:
        result["corrupt_imgs"].append((idx, img_path.name, "empty file (0 bytes)"))
        return

    if file_size < MIN_FILE_SIZE:
        result["size_outliers"].append((idx, img_path.name, file_size, "suspiciously small"))
    elif file_size > MAX_FILE_SIZE:
        result["size_outliers"].append((idx, img_path.name, file_size, "suspiciously large"))

    file_hash = ("file", hashlib.md5(raw, usedforsecurity=False).digest())
    byte_duplicate = file_hash in view_hashes
    if byte_duplicate:
        result["duplicate_views"].append((idx, img_path.name, view_hashes[file_hash]))
    else:
        view_hashes[file_hash] = img_path.name

    # Decode to detect truncation; a complete JPEG may have trailing bytes.
    is_jpeg = False
    try:
        with Image.open(io.BytesIO(raw)) as img:
            is_jpeg = img.format == "JPEG"
            img.load()
            if not is_jpeg:
                actual_format = img.format or "unknown"
                result["corrupt_imgs"].append((
                    idx, img_path.name, f"expected JPEG, got {actual_format}",
                ))
                return
            # Pillow can recover damaged JPEG scans without raising. Decode
            # every component again with recoverable errors treated as fatal.
            try:
                simplejpeg.decode_jpeg(
                    raw, colorspace="CMYK" if img.mode == "CMYK" else "RGB",
                    strict=True,
                )
            except ValueError as exc:
                result["corrupt_imgs"].append((idx, img_path.name, f"strict JPEG: {exc}"))
                return
            pixels = np.asarray(img)
            if img.mode == "RGB":
                pixel_data = pixels if pixels.flags.c_contiguous else np.ascontiguousarray(pixels)
                pixel_hash = ("pixels", img.mode, img.size, hashlib.sha256(pixel_data).digest())
                if pixel_hash in view_hashes and not byte_duplicate:
                    result["duplicate_views"].append((idx, img_path.name, view_hashes[pixel_hash]))
                view_hashes.setdefault(pixel_hash, img_path.name)
            _check_decoded_image(img, idx, img_path.name, result, pixels)
    except Exception as e:
        if is_jpeg and isinstance(e, OSError) and "truncated" in str(e).lower():
            result["truncated_imgs"].append((idx, img_path.name))
        result["corrupt_imgs"].append((idx, img_path.name, str(e)))


def _check_decoded_image(img, idx, name, result, pixels=None):
    """Check dimensions, color mode, and spatial detail after a full decode."""
    w, h = img.size
    if (w, h) != (1024, 1024):
        result["bad_dimensions"].append((idx, name, f"{w}x{h}"))
    if img.mode != "RGB":
        result["bad_color_mode"].append((idx, name, img.mode))
    if (w, h) == (1024, 1024) and img.mode == "RGB":
        result["images_ok"] += 1

    # Keep the uint8 buffer until blur detection needs float32 pixels.
    if pixels is None:
        pixels = np.asarray(img)
    # A view is blank only if EVERY channel has little spatial detail. Pillow's
    # histogram statistics are mathematically equivalent for JPEG's integer
    # image modes and avoid NumPy's much slower full float64 reduction.
    std_val = _max_channel_std(img)
    if std_val < BLANK_STD_THRESHOLD:
        result["blank_imgs"].append((idx, name, round(std_val, 2)))
        return

    blur_score = _image_blur_score(img, pixels)
    if blur_score < BLUR_THRESHOLD:
        result["blurry_imgs"].append((idx, name, round(blur_score, 2)))
    elif img.mode == "RGB" and (w, h) == (1024, 1024):
        side = _image_regional_blur(pixels)
        if side is not None:
            result["regional_blurry_imgs"].append((idx, name, side))


def _image_regional_blur(pixels):
    """Find a large low-detail side beside sharp scenery in a 1024x1024 RGB view.

    These conservative thresholds reproduce regional v2 from the photo review.
    Exclude the upper sky region and require both weak absolute detail/contrast
    and a strong left/right difference. This is not a general blur classifier.
    """
    small = _average_pool_rgb_4x4(pixels)
    gray = small @ np.array([_BT601_R, _BT601_G, _BT601_B], dtype=np.float32)
    cells = gray.reshape(8, 32, 8, 32).transpose(0, 2, 1, 3)
    color_cells = small.reshape(8, 32, 8, 32, 3).transpose(0, 2, 1, 3, 4)
    color_std = color_cells.std(axis=(2, 3)).max(axis=2)
    laplacian = (cells[..., :-2, 1:-1] + cells[..., 2:, 1:-1]
                 + cells[..., 1:-1, :-2] + cells[..., 1:-1, 2:]
                 - 4 * cells[..., 1:-1, 1:-1])
    focus = laplacian.var(axis=(2, 3))
    for side, edge, opposite in (("left", slice(0, 2), slice(-2, None)),
                                 ("right", slice(-2, None), slice(0, 2))):
        detail = focus[2:, edge]
        median = float(np.median(detail))
        other = float(np.median(focus[2:, opposite]))
        if (median < 8 and np.percentile(detail, 75) < 15
                and other > 100 and other / (median + .1) > 50
                and np.median(color_std[2:, edge]) < 8):
            return side
    return None


def _max_channel_std(img):
    """Return the largest per-channel population standard deviation."""
    return float(max(ImageStat.Stat(img).stddev))


def _average_pool_rgb_4x4(pixels):
    """Average 4x4 RGB blocks exactly while avoiding a full-size float copy."""
    small = np.zeros((256, 256, 3), dtype=np.float32)
    for row_offset in range(4):
        for col_offset in range(4):
            small += pixels[row_offset::4, col_offset::4]
    small *= 1.0 / 16.0
    return small


def _image_blur_score(img, pixels):
    """Compute grayscale Laplacian variance using the already decoded pixels."""
    if pixels.ndim == 3 and pixels.shape[2] == 3:
        rgb = pixels.astype(np.float32)
        gray = _BT601_R * rgb[:, :, 0] + _BT601_G * rgb[:, :, 1] + _BT601_B * rgb[:, :, 2]
    elif pixels.ndim == 2:
        gray = pixels.astype(np.float32)
    else:
        gray = np.array(img.convert("L"), dtype=np.float32)  # RGBA/CMYK/etc.
    laplacian = (
        gray[:-2, 1:-1] + gray[2:, 1:-1]
        + gray[1:-1, :-2] + gray[1:-1, 2:]
        - 4 * gray[1:-1, 1:-1]
    )
    return float(laplacian.var())


def _check_images(folder, idx, result):
    """Check all images in one filesystem location folder."""
    members = [entry for entry in folder.iterdir() if entry.is_file()]
    _check_image_members(members, idx, result, lambda path: path.read_bytes())


def _archive_member_bytes(member):
    """Read an archive member using the current worker's open file handle."""
    if _archive_file is None:
        raise OSError("archive worker is not initialized")
    return _read_archive_bytes(_archive_file, member)


def _check_metadata(meta_path, idx, source_coords, completed_panoids, result):
    """Check one metadata JSON file. Mutates result in place."""
    if not meta_path.exists():
        result["missing_meta"] = True
        return

    try:
        meta_raw = meta_path.read_bytes()
        result["disk_bytes"] += len(meta_raw)
        meta = json.loads(meta_raw)

        _check_metadata_fields(meta, idx, result)
        _check_metadata_provenance(meta, idx, result)
        _check_metadata_csv_matches(meta, idx, source_coords, completed_panoids, result)
        _check_metadata_coordinates(meta, idx, result)

    except Exception as e:
        result["corrupt_meta"] = (idx, str(e))


def _check_metadata_fields(meta, idx, result):
    """Validate required fields, scraper configuration, and location identity."""
    missing_fields = REQUIRED_META_FIELDS - set(meta.keys())
    if missing_fields:
        result["meta_field_issues"] = (idx, sorted(missing_fields))

    if "headings" in meta and meta["headings"] != EXPECTED_HEADINGS:
        result["meta_value_issues"].append((idx, "headings", EXPECTED_HEADINGS, meta["headings"]))
    if "view_resolution" in meta and meta["view_resolution"] != EXPECTED_VIEW_RESOLUTION:
        result["meta_value_issues"].append((idx, "view_resolution", EXPECTED_VIEW_RESOLUTION, meta["view_resolution"]))
    if "view_fov" in meta:
        try:
            if float(meta["view_fov"]) != EXPECTED_VIEW_FOV:
                result["meta_value_issues"].append((idx, "view_fov", EXPECTED_VIEW_FOV, meta["view_fov"]))
        except (ValueError, TypeError):
            result["meta_value_issues"].append((idx, "view_fov", EXPECTED_VIEW_FOV, meta["view_fov"]))

    if "index" in meta:
        try:
            if _parse_index(meta["index"]) != idx:
                result["meta_index_mismatch"] = (idx, meta["index"])
        except (TypeError, ValueError):
            result["meta_index_mismatch"] = (idx, meta["index"])

    if "panoid" in meta:
        panoid = meta["panoid"]
        if isinstance(panoid, str) and panoid.strip():
            result["panoid"] = panoid
        else:
            result["meta_value_issues"].append((idx, "panoid", "nonempty string", panoid))

    for coord_key in ("pano_lat", "pano_lon", "original_lat", "original_lon"):
        if coord_key in meta and meta[coord_key] is None:
            result["meta_value_issues"].append((idx, coord_key, "numeric coordinate", None))


def _check_metadata_provenance(meta, idx, result):
    """Check panorama copyright, country, and capture date."""
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
    if date_val:
        date_text = str(date_val)
        try:
            if not _DATE_RE.fullmatch(date_text):
                raise ValueError
            date.fromisoformat(f"{date_text}-01")
        except ValueError:
            result["bad_dates"].append((idx, date_text))


def _check_metadata_csv_matches(meta, idx, source_coords, completed_panoids, result):
    """Compare panorama identity and requested coordinates with CSV records."""
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
            try:
                parsed_lat = float(meta_lat)
                parsed_lon = float(meta_lon)
            except (ValueError, TypeError):
                # Coordinate validation reports malformed values. Do not turn
                # otherwise parseable JSON into a generic corruption finding.
                return
            if (abs(parsed_lat - src_lat) > COORD_TOLERANCE
                    or abs(parsed_lon - src_lon) > COORD_TOLERANCE):
                result["coord_mismatch"] = (
                    idx, parsed_lat, parsed_lon, src_lat, src_lon,
                )


def _check_metadata_coordinates(meta, idx, result):
    """Check panorama distance and geographic coordinate bounds."""
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


def _new_folder_result(idx):
    """Create the mutable result container shared by both storage backends."""
    return dict(
        idx=idx,
        empty=False,
        incomplete=None,
        corrupt_imgs=[],
        bad_dimensions=[],
        bad_color_mode=[],
        size_outliers=[],
        blank_imgs=[],
        blurry_imgs=[],
        regional_blurry_imgs=[],
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


def _process_folder(folder, has_metadata_dir, metadata_dir):
    """Process one filesystem location folder."""
    try:
        idx = int(folder.name)
    except ValueError:
        return None

    result = _new_folder_result(idx)

    try:
        _check_images(folder, idx, result)
    except Exception as e:
        result["corrupt_imgs"].append((idx, "?", f"unreadable folder: {e}"))

    if has_metadata_dir:
        meta_path = metadata_dir / f"{idx:06d}.json"
        _check_metadata(meta_path, idx, _source_coords, _completed_panoids, result)

    return result


def _process_archive_folder(folder, has_metadata_dir):
    """Process one indexed location folder inside an uncompressed tar."""
    result = _new_folder_result(folder.idx)
    try:
        _check_image_members(folder.images, folder.idx, result, _archive_member_bytes)
    except Exception as exc:
        result["corrupt_imgs"].append((folder.idx, "?", f"unreadable archive folder: {exc}"))

    if has_metadata_dir:
        if folder.metadata is None:
            result["missing_meta"] = True
        else:
            _check_metadata(
                folder.metadata, folder.idx, _source_coords, _completed_panoids, result,
            )
    return result


def _process_folder_batch(
    folders, has_metadata_dir, metadata_dir, source_coords, completed_panoids,
):
    """Process a small filesystem batch with batch-local CSV lookups."""
    _set_worker_lookups(source_coords, completed_panoids)
    return [
        _process_folder(folder, has_metadata_dir, metadata_dir) for folder in folders
    ]


def _process_archive_batch(
    folders, has_metadata_dir, source_coords=_UNSET, completed_panoids=_UNSET,
):
    """Process a small archive-order batch in one worker invocation."""
    if source_coords is not _UNSET or completed_panoids is not _UNSET:
        _set_worker_lookups(
            _source_coords if source_coords is _UNSET else source_coords,
            _completed_panoids if completed_panoids is _UNSET else completed_panoids,
        )
    return [_process_archive_folder(folder, has_metadata_dir) for folder in folders]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t0 = time.time()
    args = _parse_args()

    data_path = Path(args.data_dir)
    diagnostics = InputDiagnostics()

    print("SALTY Integrity Check")
    print("=" * 60)
    try:
        if not data_path.exists():
            raise InputValidationError(f"{data_path} not found")

        is_archive = data_path.is_file()
        if is_archive and data_path.suffix.lower() != ".tar":
            raise InputValidationError(
                f"{data_path} is not an uncompressed .tar archive"
            )

        if is_archive:
            print(f"Archive: {data_path.resolve()}")
            archive_data = _index_tar_archive(data_path)
            print(f"Archive root: {archive_data.root_name}")
            has_metadata_dir = archive_data.has_metadata_dir
            if not has_metadata_dir:
                diagnostics.warn(
                    "metadata/ not found in archive — metadata checks will be skipped"
                )
            folders = archive_data.folders
            disk_indices = archive_data.disk_indices
            metadata_source = archive_data.metadata_indices
        else:
            salty_data = data_path
            images_dir = salty_data / "images"
            metadata_dir = salty_data / "metadata"
            print(f"Directory: {salty_data.resolve()}")
            has_metadata_dir = _validate_dirs(salty_data, images_dir, diagnostics)
            folders, disk_indices = _collect_folders(images_dir)
            metadata_source = metadata_dir

        print()
        print("Loading CSVs...")
        if is_archive:
            csv_data = _load_csv_file_lists(
                archive_data.completed_files, archive_data.rejected_files, diagnostics,
            )
        else:
            csv_data = _load_csvs(salty_data, diagnostics)

        source_coords, source_indices = None, None
        if args.source_csv:
            source_coords, source_indices = _load_source_csv(
                Path(args.source_csv), diagnostics,
            )

        if (not disk_indices and not csv_data["completed_set"]
                and not csv_data["rejected_set"]):
            raise InputValidationError(
                "dataset contains no numeric image folders and no completed or rejected records"
            )
    except (ValueError, OSError, tarfile.TarError) as exc:
        print(f"ERROR: {exc}")
        print("Input validation failed; no scan was run and flagged.txt was not changed.")
        sys.exit(2)

    print(f"\nScanning {len(folders):,} image folders with {args.workers} workers...")

    if is_archive:
        findings = _scan_archive_folders(
            data_path, folders, has_metadata_dir,
            source_coords, csv_data["completed_panoids"], args.workers,
        )
    else:
        findings = _scan_all_folders(
            folders, has_metadata_dir, metadata_dir,
            source_coords, csv_data["completed_panoids"], args.workers,
        )
    derived = _compute_derived(
        findings, csv_data, disk_indices, has_metadata_dir, metadata_source, source_indices,
    )

    issues, warnings = _print_report(
        findings, derived, disk_indices, has_metadata_dir,
        len(csv_data["rejected_set"]), csv_data["reject_reasons"], source_indices,
        diagnostics.warning_count,
    )

    elapsed = time.time() - t0
    print()
    print("=" * 60)
    if issues:
        print(f"{issues:,} failure(s); {warnings:,} warning(s) found.")
    elif warnings:
        print(f"No failures; {warnings:,} warning(s) found.")
    else:
        print("All checks passed.")
    print(f"Completed in {elapsed:.1f}s")

    if not args.no_export:
        sections     = _build_export_sections(findings, derived)
        flagged_path = data_path.parent / "flagged.txt" if is_archive else data_path / "flagged.txt"
        n_flagged    = write_flagged_export(flagged_path, sections)
        if n_flagged > 0:
            print(f"Flagged export: {flagged_path.name} ({n_flagged:,} entries)")
        else:
            print("Flagged export: nothing to flag")

    print()
    sys.exit(0 if issues == 0 else 1)


if __name__ == "__main__":
    main()
