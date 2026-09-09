"""Summarize saved measurements and visual labels without rescanning the datasets."""

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from evaluate import regional_v2

HERE = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def with_current_rule(record):
    record = dict(record)
    if "focus_grid" in record:
        record["regional_v2"] = regional_v2(
            np.array(record["focus_grid"]), np.array(record["std_grid"]))
    return record


def confusion(labels, records, predict):
    counts = dict(TP=0, FP=0, FN=0, TN=0)
    for label in labels:
        if label["label"] not in ("degraded", "usable"):
            continue
        actual = label["label"] == "degraded"
        predicted = bool(predict(records[label["path"]]))
        counts[("T" if predicted == actual else "F") + ("P" if predicted else "N")] += 1
    return counts


def main():
    records = {r["path"]: with_current_rule(r) for r in read(HERE / "results/metrics.json")}
    development = read(HERE / "labels.json")
    validation = read(HERE / "validation_labels.json")
    audit = read(HERE / "candidate_labels.json")
    baseline = lambda r: bool(r.get("baseline"))
    methods = {"baseline": baseline}
    for key in ("naive_regional", "regional", "regional_v2", "near_blank"):
        methods[key] = lambda r, key=key: r.get(key, False)
    methods["baseline_plus_regional_v2"] = lambda r: baseline(r) or r.get("regional_v2", False)
    methods["baseline_plus_both_visual_rules"] = lambda r: (
        baseline(r) or r.get("regional_v2", False) or r.get("near_blank", False))

    result = {"note": "Exploratory evaluation, with agent visual labels. Candidate audit is selected from predictions; not an independent accuracy estimate. Regional v2 is recomputed from saved grids rounded to three decimals.",
              "images": len(records), "sources": dict(Counter(r["source"] for r in records.values()))}
    result["labeled_sets"] = {
        name: {"labels": dict(Counter(r["label"] for r in labels)),
               "methods": {method: confusion(labels, records, predict)
                           for method, predict in methods.items()}}
        for name, labels in (("development", development), ("validation", validation))}
    result["candidate_audit"] = {}
    folder_records = defaultdict(list)
    for record in records.values():
        folder_records[str(Path(record["path"]).parent)].append(record)
    for candidate in ("regional_v2", "near_blank"):
        flagged = [r for r in records.values() if r.get(candidate)]
        labels = [r for r in audit if r["candidate"] == candidate]
        if {r["path"] for r in flagged} != {r["path"] for r in labels}:
            raise ValueError(f"Visual audit does not cover the exact {candidate} candidate set")
        folders = {str(Path(r["path"]).parent) for r in flagged}
        result["candidate_audit"][candidate] = {
            "flagged": len(flagged),
            "labels": dict(Counter(r["label"] for r in labels)),
            "individual_images_passing_baseline": sum(not baseline(r) for r in flagged),
            "passing_baseline_and_visually_degraded": sum(
                r["label"] == "degraded" and not baseline(records[r["path"]]) for r in labels),
            "distinct_dataset_folders": len(folders),
            "folders_with_baseline_issue_in_sample": sum(
                any(baseline(r) for r in folder_records[folder]) for folder in folders),
            "random_active_flags": sum(r["source"] == "random_active" for r in flagged),
            "counterexamples": [r for r in labels if r["label"] != "degraded"],
            "paths": [r["path"] for r in flagged]}

    groups = defaultdict(list)
    for record in records.values():
        if record.get("pixel_sha256"):
            groups[(str(Path(record["path"]).parent), record["pixel_sha256"])].append(record)
    duplicate_groups = [group for group in groups.values() if len(group) > 1]
    result["decoded_duplicates"] = {
        "scope": "Only sampled views within the same dataset/location directory",
        "groups": [[r["path"] for r in group] for group in duplicate_groups]}
    issues = [r for r in records.values() if r.get("strict", {}).get("status") == "issue"]
    result["strict_jpeg"] = {
        "statuses": dict(Counter(r.get("strict", {}).get("status", "not_run") for r in records.values())),
        "additional_issues": [r["path"] for r in issues if not r.get("baseline", {}).get("corrupt_imgs")],
        "messages": dict(Counter(r["strict"]["message"] for r in issues))}
    result["timing_ms_valid_files"] = {}
    for key in ("baseline_ms", "strict_ms", "pixel_hash_ms", "features_ms"):
        values = [r[key] for r in records.values()
                  if key in r and r.get("strict", {}).get("status") == "ok"]
        result["timing_ms_valid_files"][key] = {
            "median": round(float(np.median(values)), 3),
            "p95": round(float(np.percentile(values, 95)), 3)}
    synthetic = read(HERE / "results/synthetic.json")
    result["synthetic"] = {
        "source": synthetic["source"], "duplicates": synthetic["duplicate_results"],
        "cases": {name: {"baseline": list(r["baseline"]),
                         "strict": r["strict"],
                         "regional_v2": with_current_rule(r).get("regional_v2"),
                         "near_blank": r.get("near_blank")}
                  for name, r in synthetic["cases"].items()}}
    output = HERE / "results/evaluation.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {output}")
    print(json.dumps({key: result[key] for key in ("labeled_sets", "strict_jpeg", "timing_ms_valid_files")}, indent=2))


if __name__ == "__main__":
    main()
