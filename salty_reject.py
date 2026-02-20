"""
SALTY Reject Tool
Moves flagged entries from completed → rejected with full archive-based undo.

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
from datetime import datetime
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

def remove_from_csv(csv_path, idx):
    """Remove all rows with the given index from a CSV. Returns number of rows removed."""
    try:
        df = pd.read_csv(csv_path)
        mask = pd.to_numeric(df["index"], errors="coerce") == idx
        n = int(mask.sum())
        if n > 0:
            # Write to temp file first, then atomically replace — protects against
            # corrupted CSV if the process is interrupted mid-write
            tmp = csv_path.with_suffix(".tmp")
            df[~mask].to_csv(tmp, index=False)
            os.replace(tmp, csv_path)
        return n
    except Exception as e:
        print(f"  WARNING: Could not update {csv_path.name}: {e}")
        return 0


def append_to_csv(csv_path, row_dict):
    """Append a row dict to a CSV (creates with header if file doesn't exist).
    Note: uses direct append mode (not atomic). A crash mid-write could leave a
    truncated row. Recovery records ensure undo still works in that case.
    """
    df = pd.DataFrame([row_dict])
    if csv_path.exists():
        df.to_csv(csv_path, mode="a", header=False, index=False)
    else:
        df.to_csv(csv_path, mode="w", header=True, index=False)


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
# Reject mode
# ---------------------------------------------------------------------------

def do_reject(data_dir, index_reasons, dry_run):
    """Move entries from completed → archive + rejects."""
    IMAGES_DIR    = data_dir / "images"
    METADATA_DIR  = data_dir / "metadata"
    ARCHIVE_DIR   = data_dir / "rejected_archive"
    ARCH_IMAGES   = ARCHIVE_DIR / "images"
    ARCH_META     = ARCHIVE_DIR / "metadata"
    ARCH_RECORDS  = ARCHIVE_DIR / "records"

    completed_files = sorted(data_dir.glob("completed*.csv"))
    rejected_files  = sorted(data_dir.glob("rejects*.csv"))
    rejects_csv     = rejected_files[0] if rejected_files else data_dir / "rejects.csv"

    # Pre-load once — avoids re-reading multi-MB CSVs for every rejected index
    rejected_set = set()
    for rf in rejected_files:
        try:
            df = pd.read_csv(rf)
            rejected_set.update(pd.to_numeric(df["index"], errors="coerce").dropna().astype(int))
        except Exception:
            pass

    # Build index → (row_dict, source_path) map for O(1) per-entry lookup.
    # pd.to_numeric on a 105k-row series inside the loop would be O(N) per index.
    completed_index = {}
    for f in completed_files:
        try:
            df = pd.read_csv(f)
            idx_col = pd.to_numeric(df["index"], errors="coerce")
            records = df.to_dict("records")
            for idx_val, record in zip(idx_col, records):
                if pd.notna(idx_val):
                    idx_int = int(idx_val)
                    if idx_int not in completed_index:
                        completed_index[idx_int] = (
                            {k: (None if pd.isna(v) else v)
                             for k, v in record.items()
                             if not k.startswith("Unnamed:")},
                            f,
                        )
        except Exception as e:
            print(f"  WARNING: Could not read {f.name}: {e}")

    has_metadata_dir = METADATA_DIR.exists()

    if not dry_run:
        ARCH_IMAGES.mkdir(parents=True, exist_ok=True)
        ARCH_META.mkdir(parents=True, exist_ok=True)
        ARCH_RECORDS.mkdir(parents=True, exist_ok=True)

    n_rejected = n_skipped = n_not_found = 0

    for idx, reason in index_reasons:
        idx_str = f"{idx:06d}"

        # Already rejected?
        if idx in rejected_set:
            print(f"  SKIP {idx_str}: already in rejects")
            n_skipped += 1
            continue

        # Partial failure from a previous run?
        record_path = ARCH_RECORDS / f"{idx_str}.json"
        if record_path.exists():
            print(f"  SKIP {idx_str}: recovery record already exists — "
                  f"previous run may have failed mid-way. Use --undo first.")
            n_skipped += 1
            continue

        # Determine what exists
        completed_row, source_file = completed_index.get(idx, (None, None))
        images_folder = IMAGES_DIR / idx_str
        meta_json     = METADATA_DIR / f"{idx_str}.json"
        has_completed = completed_row is not None
        has_images    = images_folder.exists()
        has_metadata  = has_metadata_dir and meta_json.exists()

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

        # Gather lat/lon/panoid for rejects.csv
        lat, lon, panoid = 0.0, 0.0, "N/A"
        if has_metadata:
            try:
                meta_data = json.loads(meta_json.read_text(encoding="utf-8"))
                lat  = float(meta_data.get("original_lat", 0.0) or 0.0)
                lon  = float(meta_data.get("original_lon", 0.0) or 0.0)
                panoid = str(meta_data.get("panoid", "N/A") or "N/A")
            except Exception:
                pass
        if lat == 0.0 and lon == 0.0 and has_completed:
            lat = float(completed_row.get("lat", 0.0) or 0.0)
            lon = float(completed_row.get("lon", 0.0) or 0.0)
        if panoid == "N/A" and has_completed:
            panoid = str(completed_row.get("panoid", "N/A") or "N/A")
        if lat == 0.0 and lon == 0.0:
            print(f"  WARN {idx_str}: could not determine lat/lon, using 0.0")

        if dry_run:
            print(f"  [DRY RUN] {idx_str}")
            if has_completed:
                print(f"    remove from {source_file.name}")
            print(f"    add to {rejects_csv.name}  (reason: {reason})")
            if has_images:
                print(f"    archive: images/{idx_str}/ → rejected_archive/images/{idx_str}/")
            if has_metadata:
                print(f"    archive: metadata/{idx_str}.json → rejected_archive/metadata/{idx_str}.json")
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
                dest = ARCH_IMAGES / idx_str
                if dest.exists():
                    print(f"  WARN {idx_str}: archive destination {dest} already exists — skipping image archive")
                else:
                    shutil.move(str(images_folder), str(dest))
            if has_metadata:
                dest = ARCH_META / f"{idx_str}.json"
                if dest.exists():
                    print(f"  WARN {idx_str}: archive destination {dest} already exists — skipping metadata archive")
                else:
                    shutil.move(str(meta_json), str(dest))
            # 3. Update CSVs
            if has_completed:
                remove_from_csv(source_file, idx)
            try:
                append_to_csv(rejects_csv, rejects_row)
            except Exception as e:
                print(f"  ERROR {idx_str}: could not write to {rejects_csv.name}: {e}")
                print(f"    Recovery record preserved — run --undo to restore this entry.")
                n_skipped += 1
                continue

        n_rejected += 1

    print()
    prefix = "[DRY RUN] " if dry_run else ""
    print(f"{prefix}Summary: {n_rejected} {'to reject' if dry_run else 'rejected'}, "
          f"{n_skipped} skipped, {n_not_found} not found")


# ---------------------------------------------------------------------------
# Undo mode
# ---------------------------------------------------------------------------

def do_undo(data_dir, index_reasons, dry_run):
    """Restore archived entries back to completed."""
    IMAGES_DIR   = data_dir / "images"
    METADATA_DIR = data_dir / "metadata"
    ARCHIVE_DIR  = data_dir / "rejected_archive"
    ARCH_IMAGES  = ARCHIVE_DIR / "images"
    ARCH_META    = ARCHIVE_DIR / "metadata"
    ARCH_RECORDS = ARCHIVE_DIR / "records"

    rejected_files = sorted(data_dir.glob("rejects*.csv"))

    n_restored = n_skipped = n_not_found = 0

    for idx, _ in index_reasons:
        idx_str = f"{idx:06d}"
        record_path = ARCH_RECORDS / f"{idx_str}.json"

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

        had_images       = record.get("had_images", False)
        had_metadata     = record.get("had_metadata", False)
        had_completed    = record.get("had_completed_row", False)
        completed_row    = record.get("completed_row")
        source_file_name = record.get("completed_source_file")

        arch_img  = ARCH_IMAGES / idx_str
        arch_meta = ARCH_META / f"{idx_str}.json"

        if dry_run:
            print(f"  [DRY RUN] {idx_str}")
            if had_images:
                print(f"    restore: rejected_archive/images/{idx_str}/ → images/{idx_str}/")
            if had_metadata:
                print(f"    restore: rejected_archive/metadata/{idx_str}.json → metadata/{idx_str}.json")
            if had_completed and completed_row:
                print(f"    add back to {source_file_name or 'completed.csv'}")
            print("    remove from rejects")
        else:
            # Restore image folder
            if had_images:
                if arch_img.exists():
                    dest = IMAGES_DIR / idx_str
                    if dest.exists():
                        print(f"  WARN {idx_str}: restore destination {dest} already exists — skipping image restore")
                    else:
                        IMAGES_DIR.mkdir(parents=True, exist_ok=True)
                        shutil.move(str(arch_img), str(dest))
                else:
                    print(f"  WARN {idx_str}: archive image folder not found (purged?)")

            # Restore metadata
            if had_metadata:
                if arch_meta.exists():
                    dest = METADATA_DIR / f"{idx_str}.json"
                    if dest.exists():
                        print(f"  WARN {idx_str}: restore destination {dest} already exists — skipping metadata restore")
                    else:
                        METADATA_DIR.mkdir(parents=True, exist_ok=True)
                        shutil.move(str(arch_meta), str(dest))
                else:
                    print(f"  WARN {idx_str}: archive metadata JSON not found (purged?)")

            # Remove from rejects
            for rf in rejected_files:
                remove_from_csv(rf, idx)

            # Restore completed row
            if had_completed and not completed_row:
                print(f"  WARN {idx_str}: recovery record has had_completed_row=true but no row data — skipping CSV restore")
            elif had_completed and completed_row:
                target = data_dir / source_file_name if source_file_name else data_dir / "completed.csv"
                if not target.exists():
                    print(f"  WARN {idx_str}: original CSV {source_file_name!r} not found, "
                          f"writing to completed.csv instead")
                    target = data_dir / "completed.csv"
                try:
                    append_to_csv(target, completed_row)
                except Exception as e:
                    print(f"  ERROR {idx_str}: could not write to {target.name}: {e}")
                    print(f"    Recovery record preserved — fix the issue and re-run --undo.")
                    n_skipped += 1
                    continue

            # Delete record
            record_path.unlink()

        n_restored += 1

    print()
    prefix = "[DRY RUN] " if dry_run else ""
    print(f"{prefix}Summary: {n_restored} {'to restore' if dry_run else 'restored'}, "
          f"{n_skipped} skipped, {n_not_found} not found")


# ---------------------------------------------------------------------------
# Purge mode
# ---------------------------------------------------------------------------

def do_purge(data_dir, dry_run):
    """Permanently delete all archived files (irreversible)."""
    ARCHIVE_DIR  = data_dir / "rejected_archive"

    if not ARCHIVE_DIR.exists():
        print("rejected_archive/ does not exist — nothing to purge")
        return

    arch_images  = ARCHIVE_DIR / "images"
    arch_records = ARCHIVE_DIR / "records"
    n_folders = sum(1 for p in arch_images.iterdir() if p.is_dir()) if arch_images.exists() else 0
    n_records = sum(1 for p in arch_records.iterdir() if p.suffix == ".json") if arch_records.exists() else 0

    print("Purge would permanently delete:")
    print(f"  {n_folders:,} archived image folders")
    print(f"  {n_records:,} recovery records")
    print("WARNING: After purge, rejected entries cannot be undone.")

    if dry_run:
        print("[DRY RUN] No changes made.")
        return

    confirm = input("Type 'PURGE' to confirm permanent deletion: ")
    if confirm != "PURGE":
        print("Cancelled.")
        return

    shutil.rmtree(str(ARCHIVE_DIR))
    print(f"Deleted {ARCHIVE_DIR}")


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
