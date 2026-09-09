"""Read-only image-quality experiment; never invokes the checker CLI or rejects data.

Run with the existing environment (see README.md). Results are written only to --out.
Thresholds are exploratory, not approved production defaults.
"""

import argparse
import ctypes as ct
import hashlib
import io
import json
import os
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import salty_check as checker


class StrictJPEG:
    """Experimental binding to the documented TurboJPEG API, with recovery fatal.

    Uses RGB output so all three color components are decoded. No native library
    is installed or downloaded by this script. Supply its path explicitly.
    """

    def __init__(self, library):
        self.lib = ct.CDLL(str(library))
        declarations = {
            "tjInitDecompress": ([], ct.c_void_p),
            "tjDestroy": ([ct.c_void_p], ct.c_int),
            "tjDecompressHeader3": ([ct.c_void_p, ct.c_char_p, ct.c_ulong]
                + [ct.POINTER(ct.c_int)] * 4, ct.c_int),
            "tjDecompress2": ([ct.c_void_p, ct.c_char_p, ct.c_ulong, ct.c_void_p]
                + [ct.c_int] * 5, ct.c_int),
            "tjGetErrorStr2": ([ct.c_void_p], ct.c_char_p),
            "tjGetErrorCode": ([ct.c_void_p], ct.c_int),
        }
        for name, (args, result) in declarations.items():
            fn = getattr(self.lib, name)
            fn.argtypes, fn.restype = args, result
        self.handle = self.lib.tjInitDecompress()
        if not self.handle:
            raise RuntimeError("Could not initialize TurboJPEG")

    def close(self):
        if self.handle:
            self.lib.tjDestroy(self.handle)
            self.handle = None

    def error(self):
        return {"status": "issue", "code": self.lib.tjGetErrorCode(self.handle),
                "message": self.lib.tjGetErrorStr2(self.handle).decode("utf-8", "replace")}

    def check(self, raw):
        width, height, subsampling, colorspace = [ct.c_int() for _ in range(4)]
        if self.lib.tjDecompressHeader3(self.handle, raw, len(raw), ct.byref(width),
                ct.byref(height), ct.byref(subsampling), ct.byref(colorspace)):
            return self.error()
        if width.value <= 0 or height.value <= 0 or width.value * height.value > 16_000_000:
            return {"status": "skipped", "message": "Outside experimental allocation limit"}
        if colorspace.value in (3, 4):
            return {"status": "skipped", "message": "CMYK/YCCK outside RGB experiment"}
        pixels = ct.create_string_buffer(width.value * height.value * 3)
        # TJPF_RGB = 0; TJFLAG_STOPONWARNING = 8192 (official turbojpeg.h).
        code = self.lib.tjDecompress2(self.handle, raw, len(raw), pixels,
                                    width.value, 0, height.value, 0, 8192)
        return self.error() if code else {"status": "ok"}


def patches(array):
    h, w = array.shape[:2]
    if array.ndim == 2:
        return array.reshape(8, h // 8, 8, w // 8).transpose(0, 2, 1, 3)
    return array.reshape(8, h // 8, 8, w // 8, 3).transpose(0, 2, 1, 3, 4)


def laplacian(array):
    return (array[..., :-2, 1:-1] + array[..., 2:, 1:-1]
            + array[..., 1:-1, :-2] + array[..., 1:-1, 2:]
            - 4 * array[..., 1:-1, 1:-1])


def regional_v2(focus, color_std):
    """Second exploratory rule, tuned on the two development partial-blur views.

    Compare outer quarter-width strips below the upper sky region. Relative
    sharpness alone confused real textures; require weak absolute detail too.
    This intentionally does not classify wholly blurred images.
    """
    for side, other, std in [(focus[2:, :2], focus[2:, -2:], color_std[2:, :2]),
                             (focus[2:, -2:], focus[2:, :2], color_std[2:, -2:])]:
        median = float(np.median(side))
        opposite = float(np.median(other))
        if (median < 8 and np.percentile(side, 75) < 15
                and opposite > 100 and opposite / (median + .1) > 50
                and np.median(std) < 8):
            return True
    return False


def features(pixels):
    # Average pooling suppresses pixel-scale noise before evaluating scene detail.
    rgb = pixels.astype(np.float32)
    small = rgb.reshape(256, 4, 256, 4, 3).mean(axis=(1, 3))
    gray = small @ np.array([.299, .587, .114], dtype=np.float32)
    cells = patches(gray)
    color_std = patches(small).std(axis=(2, 3)).max(axis=2)
    focus = laplacian(cells).var(axis=(2, 3))
    q = np.percentile(gray, [1, 5, 10, 50, 90, 95, 99])
    low_focus = focus < 3.0
    flat = color_std < 7.0
    # Candidate A: overly broad patch count, retained as a comparison baseline.
    naive_regional = bool(low_focus.mean() >= .50)
    # Candidate B: degradation reaching well below the sky, next to detailed content.
    side_tests = []
    for side, opposite in [(focus[2:, :3], focus[2:, -3:]),
                           (focus[2:, -3:], focus[2:, :3])]:
        side_tests.append(bool((side < 3).mean() >= .75
                               and np.median(opposite) >= 20
                               and np.median(opposite) / (np.median(side) + .1) >= 15))
    near_blank = bool(flat.mean() >= .85 and
                      (q[3] < 55 or q[3] > 220 or q[4] - q[2] < 12))
    return {
        "luminance_percentiles": q.round(3).tolist(),
        "flat_fraction": round(float(flat.mean()), 5),
        "low_focus_fraction": round(float(low_focus.mean()), 5),
        "focus_grid": focus.round(3).tolist(),
        "std_grid": color_std.round(3).tolist(),
        "naive_regional": naive_regional,
        "regional": any(side_tests),
        "regional_v2": regional_v2(focus, color_std),
        "near_blank": near_blank,
    }


def analyze(raw, strict):
    result = checker._new_folder_result(1)
    start = time.perf_counter()
    checker._check_image(Path("000.jpg"), 1, result, lambda _: raw, {})
    elapsed_baseline = time.perf_counter() - start
    baseline = {k: result[k] for k in ("corrupt_imgs", "truncated_imgs", "blank_imgs",
                 "blurry_imgs", "bad_dimensions", "bad_color_mode", "size_outliers") if result[k]}
    start = time.perf_counter()
    validation = strict.check(raw)
    elapsed_strict = time.perf_counter() - start
    out = {"baseline": baseline, "strict": validation,
           "file_sha256": hashlib.sha256(raw).hexdigest(),
           "baseline_ms": elapsed_baseline * 1000, "strict_ms": elapsed_strict * 1000}
    try:
        with Image.open(io.BytesIO(raw)) as image:
            image.load()
            if image.mode != "RGB" or image.size != (1024, 1024):
                out["features_skipped"] = "Expected RGB 1024x1024"
                return out
            px = np.asarray(image)
            start = time.perf_counter()
            out["pixel_sha256"] = hashlib.sha256(b"RGB:1024x1024:" + px.tobytes()).hexdigest()
            out["pixel_hash_ms"] = (time.perf_counter() - start) * 1000
            start = time.perf_counter()
            out.update(features(px))
            out["features_ms"] = (time.perf_counter() - start) * 1000
    except (OSError, ValueError) as exc:
        out["features_skipped"] = str(exc)
    return out


def select_images(labels):
    selected = {}
    rng = random.Random(20260910)
    for root in sorted(ROOT.glob("salty_data*")):
        active = root / "images"
        if not active.is_dir():
            continue
        with os.scandir(active) as entries:
            names = sorted(e.name for e in entries if e.name.isdigit() and e.is_dir())
        for name in rng.sample(names, min(16, len(names))):
            for heading in checker.EXPECTED_IMAGE_NAMES:
                selected[active / name / heading] = "random_active"
        archived = root / "rejected_archive" / "images"
        if archived.is_dir():
            for folder in sorted(archived.iterdir()):
                if folder.is_dir():
                    for heading in checker.EXPECTED_IMAGE_NAMES:
                        selected[folder / heading] = "archived_reject"
        # The 205k batches have flagged examples in active images, without archives.
        if "205k" in root.name and (root / "flagged.txt").exists():
            text = (root / "flagged.txt").read_text(encoding="utf-8")
            indices = sorted(set(re.findall(r"^#?\s*(\d+)\s+#\s*blurry_imgs:", text, re.M)))
            for idx in rng.sample(indices, min(8, len(indices))):
                for heading in checker.EXPECTED_IMAGE_NAMES:
                    selected[active / f"{int(idx):06d}" / heading] = "flagged_active"
    for label in labels:
        selected[ROOT / label["path"]] = "visually_reviewed"
    return dict(sorted(selected.items()))


def encode(image, **kwargs):
    stream = io.BytesIO()
    image.save(stream, format="JPEG", quality=90, **kwargs)
    return stream.getvalue()


def synthetic_tests(strict):
    source = ROOT / "salty_data_300k/images/187199/000.jpg"
    raw = source.read_bytes()
    with Image.open(io.BytesIO(raw)) as image:
        original = image.convert("RGB")
    sos = raw.index(b"\xff\xda")
    entropy_start = sos + 2 + int.from_bytes(raw[sos + 2:sos + 4], "big")
    partial = np.zeros((1024, 1024, 3), dtype=np.uint8)
    partial[:128] = np.asarray(original)[:128]
    rng = np.random.default_rng(42)
    noisy = Image.fromarray(rng.integers(0, 256, (1024, 1024, 3), dtype=np.uint8))
    # Blurred region preserves scene colors, unlike a solid synthetic rectangle.
    partial_blur = original.copy()
    partial_blur.paste(original.filter(ImageFilter.GaussianBlur(24)).crop((640, 0, 1024, 1024)), (640, 0))
    cases = {"original": raw, "identical_copy": raw,
             "neighbor_view": source.with_name("090.jpg").read_bytes(),
             "trailing_bytes": raw + b"harmless trailing data",
             "comment_segment": raw[:2] + b"\xff\xfe\x00\x07hello" + raw[2:],
             "progressive": encode(original, progressive=True),
             "missing_eoi": raw[:-2], "half_file": raw[:len(raw) // 2],
             "partial_black": encode(Image.fromarray(partial)),
             "regional_blur": encode(partial_blur), "pure_noise": encode(noisy)}
    for size in (64, 1024):
        point = entropy_start + 1000
        cases[f"removed_{size}_entropy_bytes"] = raw[:point] + raw[point + size:]
    report = {name: analyze(data, strict) for name, data in cases.items()}
    duplicate_results = []
    for other in ("identical_copy", "neighbor_view", "trailing_bytes", "comment_segment", "progressive"):
        duplicate_results.append({"case": other,
            "same_bytes": raw == cases[other],
            "same_decoded_pixels": report["original"].get("pixel_sha256") == report[other].get("pixel_sha256")})
    return {"source": source.relative_to(ROOT).as_posix(), "cases": report,
            "duplicate_results": duplicate_results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--turbojpeg", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--labels", type=Path, default=Path(__file__).with_name("labels.json"))
    args = parser.parse_args()
    out = args.out.resolve()
    # Keep experimental writes out of every source dataset, including rejected archives.
    for dataset in ROOT.glob("salty_data*"):
        if dataset.is_dir() and out.is_relative_to(dataset.resolve()):
            parser.error("--out must be outside the source datasets")
    out.mkdir(parents=True, exist_ok=True)
    labels = json.loads(args.labels.read_text(encoding="utf-8"))
    strict = StrictJPEG(args.turbojpeg)
    results = []
    started = time.perf_counter()
    try:
        synthetic = synthetic_tests(strict)
        (out / "synthetic.json").write_text(json.dumps(synthetic, indent=2), encoding="utf-8")
        files = select_images(labels)
        for number, (path, source) in enumerate(files.items(), 1):
            record = {"path": path.relative_to(ROOT).as_posix(), "source": source}
            try:
                record.update(analyze(path.read_bytes(), strict))
            except OSError as exc:
                record["read_error"] = str(exc)
            results.append(record)
            if number % 100 == 0:
                print(f"Analyzed {number}/{len(files)} images", flush=True)
    finally:
        strict.close()
    (out / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    summary = {"images": len(results), "seconds": time.perf_counter() - started,
               "sources": dict(Counter(r["source"] for r in results)),
               "strict_issues": sum(r.get("strict", {}).get("status") == "issue" for r in results),
               "new_strict_issues": [r["path"] for r in results if
                   r.get("strict", {}).get("status") == "issue" and not r.get("baseline", {}).get("corrupt_imgs")],
               "candidates": {key: sum(bool(r.get(key)) for r in results)
                              for key in ("naive_regional", "regional", "regional_v2", "near_blank")}}
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
