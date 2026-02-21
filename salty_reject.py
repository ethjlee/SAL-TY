"""
SALTY Reject Tool
Moves flagged entries from completed -> rejected with full archive-based undo.

Usage:
    uv run salty_reject.py <data_dir> --from-file flagged.txt [--dry-run]
    uv run salty_reject.py <data_dir> --undo --from-file undo_list.txt [--dry-run]
    uv run salty_reject.py <data_dir> --purge [--dry-run]
"""

import argparse
import json
import os
import shutil
import sys
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
        )


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

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
            tmp = csv_path.with_suffix(".tmp")
            df[~mask].to_csv(tmp, index=False)
            os.replace(tmp, csv_path)
        return n
    except Exception as e:
        print(f"  WARNING: Could not update {csv_path.name}: {e}")
        return 0


def _batch_append_to_csv(csv_path, rows):
    """Append multiple row dicts to a CSV in a single atomic write.
    Creates the file with a header if it does not exist.
    """
    df_new = pd.DataFrame(rows)
    if csv_path.exists():
        df = pd.concat([pd.read_csv(csv_path), df_new], ignore_index=True)
    else:
        df = df_new
    tmp = csv_path.with_suffix(".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, csv_path)


def append_to_csv(csv_path, row_dict):
    """Append a single row dict to a CSV. Atomic write via temp file."""
    _batch_append_to_csv(csv_path, [row_dict])


# ---------------------------------------------------------------------------
# Input parsing
# ---------------------------------------------------------------------------

def parse_index_file(path):
    """
    Parse a file of indices (one per line).
    Format: <index>  # optional reason comment
    Returns list of (idx, reason) tuples, deduplicated (first occurrence wins).
    """
    seen = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
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

def _load_rejected_set(data_dir):
    """Load all rejected indices from rejects*.csv into a set."""
    rejected = set()
    for f in sorted(data_dir.glob("rejects*.csv")):
        try:
            df = pd.read_csv(f)
            rejected.update(
                pd.to_numeric(df["index"], errors="coerce").dropna().astype(int)
            )
        except Exception:
            pass
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
            print(f"  WARNING: Could not read {f.name}: {e}")
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


# ---------------------------------------------------------------------------
# Reject mode
# ---------------------------------------------------------------------------

def do_reject(data_dir, index_reasons, dry_run):
    """Move entries from completed -> archive + rejects."""
    paths = _Paths.from_data_dir(data_dir)
    rejected_set    = _load_rejected_set(data_dir)
    completed_index = _build_completed_index(data_dir)

    if not dry_run:
        paths.arch_images.mkdir(parents=True, exist_ok=True)
        paths.arch_meta.mkdir(parents=True, exist_ok=True)
        paths.arch_records.mkdir(parents=True, exist_ok=True)

    n_rejected = n_skipped = n_not_found = n_error = 0

    # Accumulators for batch CSV updates — O(files) instead of O(entries²)
    completed_removals: dict = {}  # Path -> set[int]
    rejects_rows: list = []

    for idx, reason in index_reasons:
        idx_str = f"{idx:06d}"

        # Already rejected?
        if idx in rejected_set:
            print(f"  SKIP {idx_str}: already in rejects")
            n_skipped += 1
            continue

        # Partial failure from a previous run?
        record_path = paths.arch_records / f"{idx_str}.json"
        if record_path.exists():
            print(f"  SKIP {idx_str}: recovery record already exists — "
                  f"previous run may have failed mid-way. Use --undo first.")
            n_skipped += 1
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
            n_not_found += 1
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

        if dry_run:
            print(f"  [DRY RUN] {idx_str}")
            if has_completed:
                print(f"    remove from {source_file.name}")
            print(f"    add to {paths.rejects_csv.name}  (reason: {reason})")
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
            record_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
            # 2. Archive files
            if has_images:
                dest = paths.arch_images / idx_str
                if dest.exists():
                    print(f"  WARN {idx_str}: archive destination {dest} already exists — skipping image archive")
                else:
                    shutil.move(str(img_folder), str(dest))
            if has_metadata:
                dest = paths.arch_meta / f"{idx_str}.json"
                if dest.exists():
                    print(f"  WARN {idx_str}: archive destination {dest} already exists — skipping metadata archive")
                else:
                    shutil.move(str(meta_json), str(dest))
            # 3. Collect for batch CSV update (executed after the loop)
            if has_completed:
                completed_removals.setdefault(source_file, set()).add(idx)
            rejects_rows.append(rejects_row)

        n_rejected += 1

    # Batch CSV updates — one read+write per CSV file instead of one per entry
    if not dry_run:
        for source_path, indices in completed_removals.items():
            _batch_remove_from_csv(source_path, indices)
        if rejects_rows:
            try:
                _batch_append_to_csv(paths.rejects_csv, rejects_rows)
            except Exception as e:
                print(f"\nERROR: could not write to {paths.rejects_csv.name}: {e}")
                print(f"Files archived and recovery records written — run --undo to restore.")
                n_error = n_rejected
                n_rejected = 0

    print()
    prefix = "[DRY RUN] " if dry_run else ""
    parts = [
        f"{n_rejected} {'to reject' if dry_run else 'rejected'}",
        f"{n_skipped} skipped",
        f"{n_not_found} not found",
    ]
    if n_error:
        parts.append(f"{n_error} errors (check recovery records)")
    print(f"{prefix}Summary: {', '.join(parts)}")


# ---------------------------------------------------------------------------
# Undo mode
# ---------------------------------------------------------------------------

def do_undo(data_dir, index_reasons, dry_run):
    """Restore archived entries back to completed."""
    paths = _Paths.from_data_dir(data_dir)
    rejected_files = sorted(data_dir.glob("rejects*.csv"))

    n_restored = n_skipped = n_not_found = n_error = n_partial = 0
    successfully_restored: list = []  # (idx, record_path) — for batch rejects removal + unlink

    for idx, _ in index_reasons:
        idx_str = f"{idx:06d}"
        record_path = paths.arch_records / f"{idx_str}.json"

        if not record_path.exists():
            print(f"  SKIP {idx_str}: no recovery record — may have been purged or never rejected via this tool")
            n_not_found += 1
            continue

        try:
            record = json.loads(record_path.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"  SKIP {idx_str}: cannot read recovery record: {e}")
            n_skipped += 1
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
            n_restored += 1
            continue

        ok = True  # tracks whether all expected restores succeeded

        # Restore image folder
        if had_images:
            if arch_img.exists():
                dest = paths.images / idx_str
                if dest.exists():
                    print(f"  WARN {idx_str}: restore destination {dest} already exists — skipping image restore")
                    ok = False
                else:
                    paths.images.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(arch_img), str(dest))
            else:
                print(f"  WARN {idx_str}: archive image folder not found (purged?)")
                ok = False

        # Restore metadata
        if had_metadata:
            if arch_meta.exists():
                dest = paths.metadata / f"{idx_str}.json"
                if dest.exists():
                    print(f"  WARN {idx_str}: restore destination {dest} already exists — skipping metadata restore")
                    ok = False
                else:
                    paths.metadata.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(arch_meta), str(dest))
            else:
                print(f"  WARN {idx_str}: archive metadata JSON not found (purged?)")
                ok = False

        # Restore completed row BEFORE removing from rejects — if this fails,
        # the entry stays in rejects (consistent state) and record is preserved.
        if had_completed and not completed_row:
            print(f"  WARN {idx_str}: recovery record has had_completed_row=true but no row data — skipping CSV restore")
            ok = False
        elif had_completed and completed_row:
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
                n_error += 1
                continue

        # Queue for batch rejects removal + record deletion (executed after loop)
        if ok:
            successfully_restored.append((idx, record_path))
            n_restored += 1
        else:
            print(f"  WARN {idx_str}: some restores were incomplete — recovery record preserved for retry")
            n_partial += 1

    # Batch remove from all rejects CSVs — O(files) instead of O(entries × files)
    if successfully_restored and not dry_run:
        indices_to_remove = {idx for idx, _ in successfully_restored}
        for rf in rejected_files:
            _batch_remove_from_csv(rf, indices_to_remove)
        for _, rp in successfully_restored:
            rp.unlink()

    print()
    prefix = "[DRY RUN] " if dry_run else ""
    parts = [
        f"{n_restored} {'to restore' if dry_run else 'restored'}",
        f"{n_skipped} skipped",
        f"{n_not_found} not found",
    ]
    if n_error:
        parts.append(f"{n_error} errors")
    if n_partial:
        parts.append(f"{n_partial} partial (see warnings)")
    print(f"{prefix}Summary: {', '.join(parts)}")


# ---------------------------------------------------------------------------
# Purge mode
# ---------------------------------------------------------------------------

def do_purge(data_dir, dry_run):
    """Permanently delete all archived files (irreversible)."""
    paths = _Paths.from_data_dir(data_dir)

    if not paths.arch.exists():
        print("rejected_archive/ does not exist — nothing to purge")
        return

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

    print("Purge would permanently delete:")
    print(f"  {n_folders:,} archived image folders")
    print(f"  {n_meta:,} archived metadata files")
    print(f"  {n_records:,} recovery records")
    print("WARNING: After purge, rejected entries cannot be undone.")

    if dry_run:
        print("[DRY RUN] No changes made.")
        return

    confirm = input("Type 'PURGE' to confirm permanent deletion: ")
    if confirm != "PURGE":
        print("Cancelled.")
        return

    shutil.rmtree(str(paths.arch))
    print(f"Deleted {paths.arch}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="SALTY reject tool")
    parser.add_argument("data_dir", help="Path to salty_data directory")
    parser.add_argument("--from-file", metavar="FILE",
                        help="File of indices to reject/undo (one per line)")
    parser.add_argument("--undo",    action="store_true",
                        help="Restore rejected entries back to completed")
    parser.add_argument("--purge",   action="store_true",
                        help="Permanently delete rejected_archive/ (irreversible)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would happen without making any changes")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"ERROR: {data_dir} not found")
        sys.exit(1)

    if args.purge:
        do_purge(data_dir, args.dry_run)
        return

    if not args.from_file:
        print("ERROR: --from-file is required for reject/undo modes")
        sys.exit(1)

    from_file = Path(args.from_file)
    if not from_file.exists():
        print(f"ERROR: {from_file} not found")
        sys.exit(1)

    index_reasons = parse_index_file(from_file)
    if not index_reasons:
        print("No indices found in file. Nothing to do.")
        return

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
            return
        print()

    if args.undo:
        do_undo(data_dir, index_reasons, args.dry_run)
    else:
        do_reject(data_dir, index_reasons, args.dry_run)


if __name__ == "__main__":
    main()
