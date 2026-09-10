"""
SALTY Reject Tool
Moves flagged entries from completed -> rejected with full archive-based undo.

Usage:
    uv run salty_reject.py <data_dir> [--dry-run]
    uv run salty_reject.py <data_dir> --reject-all-flagged [--dry-run]
    uv run salty_reject.py <data_dir> --from-file flagged.txt [--dry-run]
    uv run salty_reject.py <data_dir> --undo [--from-file undo_subset.txt] [--dry-run]
    uv run salty_reject.py <data_dir> --purge [--dry-run]

If --from-file is omitted, reject mode uses <data_dir>/flagged.txt and undo mode
uses the automatically maintained <data_dir>/reject_list.txt.
Use --reject-all-flagged to include entries with a leading '#', without editing
flagged.txt. Headers and notes are ignored; duplicate indices use the first reason.
Without this option, only uncommented entries are processed.
Add --dry-run to preview changes. Actual rejection still asks for 'yes' and
archives entries for undo. --reject-all-flagged also works with --from-file,
but cannot be combined with --undo or --purge.
Rejection archives the entire images/<index>/ folder and metadata/<index>.json,
including all views for that location, and updates its CSV record.

Each real rejection updates <data_dir>/reject_list.txt with every entry that still
has a recovery record, including entries accumulated across separate runs. Running
with --undo and no --from-file restores that entire list. Pass --from-file to undo
only a chosen subset. Successfully restored entries are removed from reject_list.txt;
failed or partial restores remain available for retry. Recovery data is stored in
<data_dir>/rejected_archive/records/<index>.json.

Exit status is 0 for success, benign no-ops, or cancellation; 1 when any requested
entry fails, is blocked, is missing, or is only partly restored; and 2 for invalid
command-line usage.
"""

import argparse
import json
import os
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Path layout
# ---------------------------------------------------------------------------

@dataclass
class _Paths:
    data:         Path
    images:       Path
    metadata:     Path
    arch:         Path
    arch_images:  Path
    arch_meta:    Path
    arch_records: Path
    rejects_csv:  Path
    reject_list:  Path

    @classmethod
    def from_data_dir(cls, data_dir: Path) -> "_Paths":
        arch = data_dir / "rejected_archive"
        return cls(
            data=data_dir,
            images=data_dir / "images",
            metadata=data_dir / "metadata",
            arch=arch,
            arch_images=arch / "images",
            arch_meta=arch / "metadata",
            arch_records=arch / "records",
            rejects_csv=data_dir / "rejects.csv",
            reject_list=data_dir / "reject_list.txt",
        )


@dataclass
class OperationResult:
    """Outcome counters and process status for one reject-tool operation."""
    succeeded: int = 0
    skipped: int = 0
    not_found: int = 0
    blocked: int = 0
    errors: int = 0
    partial: int = 0

    @property
    def exit_code(self):
        return int(bool(self.not_found or self.blocked or self.errors or self.partial))


class ProgressLoadError(RuntimeError):
    """A progress CSV could not be loaded safely before rejection."""


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

def _write_csv_atomically(csv_path, df):
    """Replace a CSV with a DataFrame using a temporary file beside it."""
    tmp = csv_path.with_suffix(".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, csv_path)


def _batch_remove_from_csv(csv_path, indices):
    """Remove all rows whose 'index' is in indices (set/list). Returns number of rows removed.
    Writes atomically via a temp file. No-op if the file does not contain any matching rows.
    """
    indices = set(indices)
    try:
        df = pd.read_csv(csv_path)
        mask = pd.to_numeric(df["index"], errors="coerce").isin(indices)
        n = int(mask.sum())
        if n > 0:
            _write_csv_atomically(csv_path, df[~mask])
        return n
    except Exception as e:
        print(f"  WARNING: Could not update {csv_path.name}: {e}")
        return None


def _batch_append_to_csv(csv_path, rows):
    """Append multiple row dicts to a CSV in a single atomic write.
    Creates the file with a header if it does not exist.
    """
    df_new = pd.DataFrame(rows)
    if csv_path.exists():
        df = pd.concat([pd.read_csv(csv_path), df_new], ignore_index=True)
    else:
        df = df_new
    _write_csv_atomically(csv_path, df)


def append_to_csv(csv_path, row_dict):
    """Append a row unless its index is already present. Return whether it was added."""
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        existing = pd.to_numeric(df["index"], errors="coerce")
        row_index = int(row_dict["index"])
        if existing.eq(row_index).any():
            return False
    _batch_append_to_csv(csv_path, [row_dict])
    return True


def _rejects_csv_for(source_file, data_dir: Path, fallback: Path) -> Path:
    """Derive the rejects CSV path mirroring a completed CSV.
    completed_35000.csv -> rejects_35000.csv
    completed.csv       -> rejects.csv
    Falls back to rejects.csv if source_file is None.
    """
    if source_file is None:
        return fallback
    suffix = source_file.stem[len("completed"):]  # e.g. "_35000" or ""
    return data_dir / f"rejects{suffix}.csv"


def _recovery_record_indices(paths):
    """Return the numeric indices that currently have recovery records."""
    indices = []
    if paths.arch_records.exists():
        for record_path in paths.arch_records.glob("*.json"):
            try:
                indices.append(int(record_path.stem))
            except ValueError:
                continue
    return sorted(set(indices))


def _refresh_reject_list(paths):
    """Rewrite the automatic reject list from the recovery records still present."""
    tmp = paths.reject_list.with_suffix(".tmp")
    try:
        contents = "".join(f"{idx}\n" for idx in _recovery_record_indices(paths))
        tmp.write_text(contents, encoding="utf-8")
        os.replace(tmp, paths.reject_list)
        return True
    except Exception as e:
        print(f"  WARNING: Could not update {paths.reject_list.name}: {e}")
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        return False


# ---------------------------------------------------------------------------
# Input parsing
# ---------------------------------------------------------------------------

def parse_index_file(path, include_commented=False):
    """
    Parse a file of indices (one per line).
    Format: <index>  # optional reason comment
    With include_commented, also accept commented indices from flagged.txt.
    Returns list of (idx, reason) tuples, deduplicated (first occurrence wins).
    """
    seen = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                if not include_commented:
                    continue
                line = line[1:].strip()
                # Only treat numeric entries as flags, not headers or notes.
                token = line.split("#", 1)[0].strip()
                if not (token.isascii() and token.isdigit()):
                    continue
            parts = line.split("#", 1)
            token = parts[0].strip()
            reason = parts[1].strip() if len(parts) > 1 else "manual_reject"
            try:
                idx = int(token)
            except ValueError:
                print(f"  WARNING: Cannot parse index from: {repr(line)}, skipping")
                continue
            if idx not in seen:
                seen[idx] = reason
    return list(seen.items())


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _archive_resource(source, dest, idx_str, resource, move_label):
    """Archive a resource, treating destination conflicts and move errors as failures."""
    if dest.exists():
        print(f"  ERROR {idx_str}: archive destination {dest} already exists — cannot archive {resource}")
        return False
    try:
        shutil.move(str(source), str(dest))
    except Exception as e:
        print(f"  ERROR {idx_str}: {move_label} move failed: {e}")
        return False
    if not dest.exists():
        print(f"  ERROR {idx_str}: {move_label} move failed — {dest} not found after move")
        return False
    return True


def _restore_resource(source, dest, idx_str, resource, archive_label):
    """Restore a resource, reporting missing archives and destination conflicts."""
    if source.exists():
        if dest.exists():
            print(f"  WARN {idx_str}: restore destination {dest} already exists — skipping {resource} restore")
            return False
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.move(str(source), str(dest))
        except Exception as e:
            print(f"  ERROR {idx_str}: could not restore {resource}: {e}")
            return False
        if dest.exists():
            return True
        print(f"  ERROR {idx_str}: {resource} restore failed — {dest} not found after move")
        return False
    if dest.exists():
        # A previous undo may have moved this resource before a later step failed.
        print(f"  INFO {idx_str}: {resource} already restored")
        return True
    print(f"  WARN {idx_str}: archive {archive_label} not found (purged?)")
    return False


def _load_rejected_set(data_dir):
    """Load all rejected indices from rejects*.csv into a set."""
    rejected = set()
    for f in sorted(data_dir.glob("rejects*.csv")):
        try:
            df = pd.read_csv(f)
            rejected.update(
                pd.to_numeric(df["index"], errors="coerce").dropna().astype(int)
            )
        except Exception as e:
            raise ProgressLoadError(f"could not read {f.name}: {e}") from e
    return rejected


def _build_completed_index(data_dir):
    """
    Build a {idx: (row_dict, source_path)} map from all completed*.csv files.
    O(1) per-entry lookup; avoids re-reading multi-MB CSVs inside the per-index loop.
    """
    index = {}
    for f in sorted(data_dir.glob("completed*.csv")):
        try:
            df = pd.read_csv(f)
            idx_col = pd.to_numeric(df["index"], errors="coerce")
            for idx_val, record in zip(idx_col, df.to_dict("records")):
                if pd.notna(idx_val):
                    idx_int = int(idx_val)
                    if idx_int not in index:
                        clean = {}
                        for k, v in record.items():
                            if k.startswith("Unnamed:"):
                                continue
                            try:
                                clean[k] = None if pd.isna(v) else v
                            except (TypeError, ValueError):
                                clean[k] = v  # non-scalar value; keep as-is
                        index[idx_int] = (clean, f)
        except Exception as e:
            raise ProgressLoadError(f"could not read {f.name}: {e}") from e
    return index


def _resolve_lat_lon_panoid(paths, idx, completed_row):
    """
    Return (lat, lon, panoid) for an entry.
    Tries the metadata JSON first, falls back to the completed row.
    Returns (0.0, 0.0, 'N/A') if nothing usable is found.
    """
    lat = lon = panoid = None

    meta_json = paths.metadata / f"{idx:06d}.json"
    if meta_json.exists():
        try:
            data = json.loads(meta_json.read_text(encoding="utf-8"))
            raw_lat = data.get("original_lat")
            raw_lon = data.get("original_lon")
            raw_pid = data.get("panoid")
            lat    = float(raw_lat) if raw_lat is not None else None
            lon    = float(raw_lon) if raw_lon is not None else None
            panoid = str(raw_pid)   if raw_pid  is not None else None
        except Exception:
            lat = lon = panoid = None

    if completed_row:
        if lat is None:
            try:
                lat = float(completed_row.get("lat") or 0.0)
            except (TypeError, ValueError):
                pass
        if lon is None:
            try:
                lon = float(completed_row.get("lon") or 0.0)
            except (TypeError, ValueError):
                pass
        if panoid is None:
            raw = completed_row.get("panoid")
            panoid = str(raw) if raw is not None else None

    return lat or 0.0, lon or 0.0, panoid or "N/A"


def _print_operation_summary(result, action, dry_run):
    """Print the common reject/undo summary from an OperationResult."""
    past_tense = "rejected" if action == "reject" else "restored"
    success_label = f"to {action}" if dry_run else past_tense
    parts = [
        f"{result.succeeded} {success_label}",
        f"{result.skipped} skipped",
    ]
    if result.blocked:
        parts.append(f"{result.blocked} blocked")
    parts.append(f"{result.not_found} not found")
    if result.errors:
        suffix = " (check recovery records)" if action == "reject" else ""
        parts.append(f"{result.errors} errors{suffix}")
    if result.partial:
        parts.append(f"{result.partial} partial (see warnings)")
    prefix = "[DRY RUN] " if dry_run else ""
    print()
    print(f"{prefix}Summary: {', '.join(parts)}")


# ---------------------------------------------------------------------------
# Reject mode
# ---------------------------------------------------------------------------

def do_reject(data_dir, index_reasons, dry_run):
    """Move entries from completed -> archive + rejects."""
    paths = _Paths.from_data_dir(data_dir)
    result = OperationResult()
    try:
        rejected_set = _load_rejected_set(data_dir)
        completed_index = _build_completed_index(data_dir)
    except ProgressLoadError as e:
        print(f"  ERROR: {e}; no changes made")
        result.errors += 1
        _print_operation_summary(result, "reject", dry_run)
        return result

    if not dry_run:
        try:
            paths.arch_images.mkdir(parents=True, exist_ok=True)
            paths.arch_meta.mkdir(parents=True, exist_ok=True)
            paths.arch_records.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            print(f"  ERROR: could not prepare rejected_archive/: {e}")
            result.errors += 1
            _print_operation_summary(result, "reject", dry_run)
            return result

    # Accumulators for batch CSV updates — O(files) instead of O(entries²)
    completed_removals: dict = {}  # Path -> set[int]
    rejects_by_source: dict = {}   # source_file (or None) -> list[dict]

    for idx, reason in index_reasons:
        idx_str = f"{idx:06d}"

        # Already rejected?
        if idx in rejected_set:
            print(f"  SKIP {idx_str}: already in rejects")
            result.skipped += 1
            continue

        # Partial failure from a previous run?
        record_path = paths.arch_records / f"{idx_str}.json"
        if record_path.exists():
            print(f"  BLOCKED {idx_str}: recovery record already exists — "
                  f"previous run may have failed mid-way. Use --undo first.")
            result.blocked += 1
            continue

        # Determine what exists
        completed_row, source_file = completed_index.get(idx, (None, None))
        img_folder = paths.images  / idx_str
        meta_json  = paths.metadata / f"{idx_str}.json"
        has_completed = completed_row is not None
        has_images    = img_folder.exists()
        has_metadata  = meta_json.exists()

        # Nothing found anywhere
        if not has_completed and not has_images and not has_metadata:
            print(f"  SKIP {idx_str}: nothing found (not in CSV, no folder, no metadata)")
            result.not_found += 1
            continue

        # Per-resource warnings
        if not has_completed:
            print(f"  WARN {idx_str}: not in any completed*.csv")
        if not has_images:
            print(f"  WARN {idx_str}: no image folder")
        if not has_metadata:
            print(f"  WARN {idx_str}: no metadata JSON")

        lat, lon, panoid = _resolve_lat_lon_panoid(paths, idx, completed_row)
        if lat == 0.0 and lon == 0.0:
            print(f"  WARN {idx_str}: could not determine lat/lon, using 0.0")

        archive_conflicts = []
        if has_images and (paths.arch_images / idx_str).exists():
            archive_conflicts.append(paths.arch_images / idx_str)
        if has_metadata and (paths.arch_meta / f"{idx_str}.json").exists():
            archive_conflicts.append(paths.arch_meta / f"{idx_str}.json")
        if archive_conflicts:
            for conflict in archive_conflicts:
                print(f"  ERROR {idx_str}: archive destination {conflict} already exists")
            print("    Resolve the archive conflict before retrying; no changes made for this entry.")
            result.errors += 1
            continue

        if dry_run:
            print(f"  [DRY RUN] {idx_str}")
            if has_completed:
                print(f"    remove from {source_file.name}")
            target_rejects = _rejects_csv_for(source_file if has_completed else None, data_dir, paths.rejects_csv)
            print(f"    add to {target_rejects.name}  (reason: {reason})")
            if has_images:
                print(f"    archive: images/{idx_str}/ -> rejected_archive/images/{idx_str}/")
            if has_metadata:
                print(f"    archive: metadata/{idx_str}.json -> rejected_archive/metadata/{idx_str}.json")
        else:
            rejects_row = {
                "timestamp": datetime.now().isoformat(),
                "index":     idx,
                "lat":       lat,
                "lon":       lon,
                "reason":    reason,
                "panoid":    panoid,
            }
            record = {
                "completed_row":         completed_row,
                "completed_source_file": source_file.name if source_file else None,
                "had_completed_row":     has_completed,
                "had_images":            has_images,
                "had_metadata":          has_metadata,
            }
            # 1. Write recovery record FIRST (before any mutation)
            try:
                record_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
            except Exception as e:
                print(f"  ERROR {idx_str}: could not write recovery record: {e}")
                result.errors += 1
                continue
            # 2. Archive files
            if has_images:
                if not _archive_resource(
                    img_folder, paths.arch_images / idx_str, idx_str,
                    "image", "image folder",
                ):
                    result.errors += 1
                    continue
            if has_metadata:
                if not _archive_resource(
                    meta_json, paths.arch_meta / f"{idx_str}.json", idx_str,
                    "metadata", "metadata",
                ):
                    result.errors += 1
                    continue
            # 3. Collect for batch CSV update (executed after the loop)
            if has_completed:
                completed_removals.setdefault(source_file, set()).add(idx)
            rejects_by_source.setdefault(source_file if has_completed else None, []).append(rejects_row)

        result.succeeded += 1

    # Batch CSV updates — one read+write per CSV file instead of one per entry
    if not dry_run:
        failed_sources = set()
        for source_path, indices in completed_removals.items():
            removed = _batch_remove_from_csv(source_path, indices)
            if removed is None or removed < len(indices):
                print(f"  ERROR: could not remove every selected row from {source_path.name}; "
                      "recovery records preserved for undo")
                failed_sources.add(source_path)
                result.errors += len(indices)
                result.succeeded -= len(indices)
        for src, rows in rejects_by_source.items():
            if src in failed_sources:
                continue
            target = _rejects_csv_for(src, data_dir, paths.rejects_csv)
            try:
                _batch_append_to_csv(target, rows)
            except Exception as e:
                print(f"\nERROR: could not write to {target.name}: {e}")
                print(f"Files archived and recovery records written — run --undo to restore.")
                result.errors += len(rows)
                result.succeeded -= len(rows)

        # Recovery records are the source of truth. Rebuilding this file here
        # makes separate reject runs accumulate into one ready-to-use reject list.
        if not _refresh_reject_list(paths):
            result.errors += 1

    _print_operation_summary(result, "reject", dry_run)
    return result


# ---------------------------------------------------------------------------
# Undo mode
# ---------------------------------------------------------------------------

def do_undo(data_dir, index_reasons, dry_run):
    """Restore archived entries back to completed."""
    paths = _Paths.from_data_dir(data_dir)
    rejected_files = sorted(data_dir.glob("rejects*.csv"))

    result = OperationResult()
    successfully_restored: list = []  # (idx, record_path) — for batch rejects removal + unlink

    for idx, _ in index_reasons:
        idx_str = f"{idx:06d}"
        record_path = paths.arch_records / f"{idx_str}.json"

        if not record_path.exists():
            print(f"  SKIP {idx_str}: no recovery record — may have been purged or never rejected via this tool")
            result.not_found += 1
            continue

        try:
            record = json.loads(record_path.read_text(encoding="utf-8"))
            if not isinstance(record, dict):
                raise ValueError("recovery record must contain a JSON object")
        except Exception as e:
            print(f"  ERROR {idx_str}: cannot read recovery record: {e}")
            result.errors += 1
            continue

        had_images    = record.get("had_images", False)
        had_metadata  = record.get("had_metadata", False)
        had_completed = record.get("had_completed_row", False)
        completed_row = record.get("completed_row")
        source_name   = record.get("completed_source_file")

        arch_img  = paths.arch_images / idx_str
        arch_meta = paths.arch_meta / f"{idx_str}.json"

        if dry_run:
            print(f"  [DRY RUN] {idx_str}")
            if had_images:
                print(f"    restore: rejected_archive/images/{idx_str}/ -> images/{idx_str}/")
            if had_metadata:
                print(f"    restore: rejected_archive/metadata/{idx_str}.json -> metadata/{idx_str}.json")
            if had_completed and completed_row:
                print(f"    add back to {source_name or 'completed.csv'}")
            print("    remove from rejects")
            result.succeeded += 1
            continue

        ok = True  # tracks whether all expected restores succeeded

        # Restore image folder
        if had_images:
            if not _restore_resource(
                arch_img, paths.images / idx_str, idx_str,
                "image", "image folder",
            ):
                ok = False

        # Restore metadata
        if had_metadata:
            if not _restore_resource(
                arch_meta, paths.metadata / f"{idx_str}.json", idx_str,
                "metadata", "metadata JSON",
            ):
                ok = False

        # Restore the completed row only after all files are safely in place.
        # The append is idempotent so interrupted undo operations can be retried.
        if had_completed and not completed_row:
            print(f"  WARN {idx_str}: recovery record has had_completed_row=true but no row data — skipping CSV restore")
            ok = False
        elif ok and had_completed and completed_row:
            target = data_dir / source_name if source_name else data_dir / "completed.csv"
            if not target.exists():
                print(f"  WARN {idx_str}: original CSV {source_name!r} not found, "
                      f"writing to completed.csv instead")
                target = data_dir / "completed.csv"
            try:
                append_to_csv(target, completed_row)
            except Exception as e:
                print(f"  ERROR {idx_str}: could not write to {target.name}: {e}")
                print(f"    Recovery record preserved — fix the issue and re-run --undo.")
                result.errors += 1
                continue

        # Queue for batch rejects removal + record deletion (executed after loop)
        if ok:
            successfully_restored.append((idx, record_path))
        else:
            print(f"  WARN {idx_str}: some restores were incomplete — recovery record preserved for retry")
            result.partial += 1

    # Batch remove from all rejects CSVs — O(files) instead of O(entries × files)
    if successfully_restored and not dry_run:
        indices_to_remove = {idx for idx, _ in successfully_restored}
        rejects_updated = True
        for rf in rejected_files:
            if _batch_remove_from_csv(rf, indices_to_remove) is None:
                rejects_updated = False
        if rejects_updated:
            for _, rp in successfully_restored:
                try:
                    rp.unlink()
                    result.succeeded += 1
                except OSError as e:
                    print(f"  ERROR: could not remove recovery record {rp.name}: {e}")
                    result.errors += 1
        else:
            print("  ERROR: could not update all rejects CSVs; recovery records preserved for retry")
            result.errors += len(successfully_restored)
        if not _refresh_reject_list(paths):
            result.errors += 1

    _print_operation_summary(result, "restore", dry_run)
    return result


# ---------------------------------------------------------------------------
# Purge mode
# ---------------------------------------------------------------------------

def do_purge(data_dir, dry_run):
    """Permanently delete all archived files (irreversible)."""
    paths = _Paths.from_data_dir(data_dir)
    result = OperationResult()

    if not paths.arch.exists():
        print("rejected_archive/ does not exist — nothing to purge")
        return result

    try:
        n_folders = (
            sum(1 for p in paths.arch_images.iterdir() if p.is_dir())
            if paths.arch_images.exists() else 0
        )
        n_meta = (
            sum(1 for p in paths.arch_meta.iterdir() if p.suffix == ".json")
            if paths.arch_meta.exists() else 0
        )
        n_records = (
            sum(1 for p in paths.arch_records.iterdir() if p.suffix == ".json")
            if paths.arch_records.exists() else 0
        )
    except OSError as e:
        print(f"ERROR: could not inspect {paths.arch}: {e}")
        result.errors += 1
        return result

    print("Purge would permanently delete:")
    print(f"  {n_folders:,} archived image folders")
    print(f"  {n_meta:,} archived metadata files")
    print(f"  {n_records:,} recovery records")
    print("WARNING: After purge, rejected entries cannot be undone.")

    if dry_run:
        print("[DRY RUN] No changes made.")
        return result

    confirm = input("Type 'PURGE' to confirm permanent deletion: ")
    if confirm != "PURGE":
        print("Cancelled.")
        return result

    try:
        shutil.rmtree(str(paths.arch))
    except OSError as e:
        print(f"ERROR: could not delete {paths.arch}: {e}")
        result.errors += 1
        return result
    result.succeeded = 1
    if not _refresh_reject_list(paths):
        result.errors += 1
    print(f"Deleted {paths.arch}")
    return result


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="SALTY reject tool")
    parser.add_argument("data_dir", help="Path to salty_data directory")
    parser.add_argument("--from-file", metavar="FILE",
                        help="File of indices to reject, or an optional undo subset (one per line)")
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--undo", action="store_true",
                            help="Restore rejected entries; defaults to the generated reject_list.txt")
    mode_group.add_argument("--purge", action="store_true",
                            help="Permanently delete rejected_archive/ (irreversible)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would happen without making any changes")
    parser.add_argument("--reject-all-flagged", action="store_true",
                        help="Reject all listed indices, including commented entries in flagged.txt")
    args = parser.parse_args()
    if args.reject_all_flagged and (args.undo or args.purge):
        parser.error("--reject-all-flagged cannot be used with --undo or --purge")

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"ERROR: {data_dir} not found")
        return 1
    if not data_dir.is_dir():
        print(f"ERROR: {data_dir} is not a directory")
        return 1

    if args.purge:
        return do_purge(data_dir, args.dry_run).exit_code

    index_reasons = None
    if not args.from_file:
        default = data_dir / ("reject_list.txt" if args.undo else "flagged.txt")
        if args.undo:
            # Recovery records remain authoritative if the generated text file
            # is missing, stale, or could not be rewritten.
            try:
                recovery_indices = _recovery_record_indices(_Paths.from_data_dir(data_dir))
            except OSError as e:
                print(f"ERROR: could not inspect recovery records: {e}")
                return 1
            if default.exists():
                print(f"No --from-file specified, using {default}")
            elif recovery_indices:
                print("No reject_list.txt found, using archived recovery records")
            else:
                print(f"ERROR: automatic reject list {default} not found")
                return 1
            index_reasons = [(idx, "manual_reject") for idx in recovery_indices]
        elif default.exists():
            print(f"No --from-file specified, using {default}")
            args.from_file = str(default)
        else:
            print(f"ERROR: default flagged list {default} not found")
            return 1

    if index_reasons is None:
        from_file = Path(args.from_file)
        if not from_file.exists():
            print(f"ERROR: {from_file} not found")
            return 1

        try:
            index_reasons = parse_index_file(
                from_file, include_commented=args.reject_all_flagged,
            )
        except OSError as e:
            print(f"ERROR: could not read {from_file}: {e}")
            return 1
    if not index_reasons:
        print("No indices found in file. Nothing to do.")
        return 0

    mode = "undo" if args.undo else "reject"
    print("SALTY Reject Tool")
    print(f"Dataset : {data_dir.resolve()}")
    print(f"Mode    : {mode}{' (dry run)' if args.dry_run else ''}")
    print(f"Indices : {len(index_reasons):,}")
    print()

    if not args.dry_run:
        confirm = input(f"About to {mode} {len(index_reasons):,} entries. Type 'yes' to confirm: ")
        if confirm.lower() != "yes":
            print("Cancelled.")
            return 0
        print()

    if args.undo:
        result = do_undo(data_dir, index_reasons, args.dry_run)
    else:
        result = do_reject(data_dir, index_reasons, args.dry_run)
    return result.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
