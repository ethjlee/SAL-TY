# Image integrity prototype review

Status: historical prototype review. The measurements below were collected before
production integration. Regional blur, decoded RGB duplicates and strict JPEG
decoding have since been implemented in `salty_check.py`; see the project README.
The production decoder is the packaged `simplejpeg` dependency. Photos, rejection
lists and archives were not changed. Re-running this experiment now compares
against the updated checker; the saved original measurements remain the baseline
for the findings below.

The strongest addition supported by the actual photos is a conservative check for
large blurred regions beside sharp scenery. Exact decoded-image duplicates and
strict JPEG decoding also close demonstrated integrity gaps. The proposed broad
near-blank rule should **not** be integrated in its current form.

## Decisions

| Candidate | Evidence | Recommendation |
| --- | --- | --- |
| Large regional blur | The revised rule found 32 visibly degraded photos that individually pass the current checker. All 32 were visually inspected. They represent 24 panoramas in 25 dataset/location directories. | Worth adding with a narrow scope: large, nearly featureless side regions adjacent to sharp scenery. Keep the existing whole-image blur check. |
| Exact decoded-image duplicates within one location | Changing a JPEG comment or appending harmless bytes changes its file hash while preserving every decoded RGB pixel. Pixel hashes caught both controlled examples and distinguished another real heading. No such duplicate group was found in the sampled real views. | Worth adding as a small deterministic improvement to the existing duplicate check. This is exact equality, not perceptual similarity. |
| JPEG corruption tolerated by ordinary decoding | Removing 64 or 1,024 entropy-coded bytes, while retaining the end marker, passed the current checker. Strict decoding rejected both. It accepted the valid original, comment, trailing-byte and progressive controls. No additional real corrupt file was found among the 915 sampled files. | Worth adding as an integrity guard, with lower observed benefit here and an explicit native-library dependency to resolve during integration. |
| Broad near-blank / severe low-contrast rule | Flagged 11 photos: 10 judged severely lacking detail and one counterexample with recognizable tunnel structure and exit. Five pass the current checker, including that counterexample and two versions of the same dark tunnel view. | Defer this rule. It can mistake naturally dark scenes for failed images. Catching a synthetic mostly-black image does not establish that the general rule is safe. |

There is an important practical limit to the regional-blur result: **all 25
affected directories already contain another sampled view flagged by the current
checker**. This demonstrates better checking of each photo, but no additional bad
location found beyond the current location-level rejection behavior in this sample.

## Actual photo examples

| Photo | Observation |
| --- | --- |
| [205k upscaled, 837818 / 000](../../salty_data_205k_upscaled/images/837818/000.jpg) | Detailed residential street on the left; a broad blurred region obscures the right. Current individual-image checks pass; regional v2 flags it. |
| [300k archive, 538830 / 180](../../salty_data_300k/rejected_archive/images/538830/180.jpg) | Broad blur on the left beside a detailed street on the right. Current checks pass; regional v2 flags it. |
| [300k archive, 312789 / 000](../../salty_data_300k/rejected_archive/images/312789/000.jpg) | A smaller but still substantial blurred side region. Current checks pass; regional v2 flags it. |
| [300k archive, 973058 / 180](../../salty_data_300k/rejected_archive/images/973058/180.jpg) | Dark tunnel with visible road, arch, structural ribs and exit. The near-blank rule flags it, which is a reason to defer that rule. |
| [205k upscaled, 1408081 / 270](../../salty_data_205k_upscaled/images/1408081/270.jpg) | Snow and sky are naturally smooth. The naive low-detail patch count flags it; regional v2 preserves it. |
| [300k, 481055 / 180](../../salty_data_300k/images/481055/180.jpg) | Natural left/right texture imbalance. Regional v2 preserves it. |

Blur in an otherwise well-formed JPEG is a photographic quality problem; these
observations do not establish whether its origin was source imagery, intentional
obscuration, downloading, or projection.

## Sample and validation limits

The run measured 915 real files across the six available `salty_data*` directories:

- 384 randomly selected active views: 16 locations, four headings each, per dataset.
- 445 other views from the 100k and 300k rejected archives.
- 64 other views from flagged active locations in the two 205k datasets.
- 22 previously selected visual development examples. These override overlapping
  selection categories, so the archive selection originally contributed 448 paths.

There were 868 decodable RGB 1024-by-1024 images and 47 files already classified as
corrupt by the checker. Strict decoding accepted those 868 and rejected those same
47; it found no additional real failure. This is not a full scan of the datasets.

Visual inspection covered **79 distinct image paths**: 22 development examples,
21 validation examples, and all 43 predictions from the final two visual rules,
with overlap removed. A rejected location was not treated as proof that every
heading was defective. Labels are the assistant's visual judgments, not user
annotations or a measurement of ViT accuracy.

The 21 validation images were labeled before comparing their predictions and share
no panorama IDs with the development set. Regional v2 was an exploratory revision
after the first rule failed, using the two development examples of partial blur.
This is not a pristine, single-shot blind benchmark. The validation set contains
no labeled regional-blur positives, so it checks counterexamples but cannot measure
regional-blur recall.

| Evaluation | Result |
| --- | --- |
| Regional v2 on the two development regional-blur examples | Both detected. |
| Regional v2 on 10 development and 15 validation photos labeled usable | No flags among these 25 counterexamples. This small selected set does not establish a general false-positive rate. |
| Regional v2 candidate audit | All 32 predictions visibly contain substantial side blur. Predictions were selected for inspection, so this audit cannot measure missed defects. |
| Regional v2 on the 384 random active views | No flags. Most were not visually labeled; this is not evidence that every one is good. |
| Near-blank rule on the same 25 usable counterexamples | No flags, but the subsequent candidate audit revealed the additional tunnel counterexample. |
| Naive low-detail patch count | Incorrectly flags a snow example and two fog/water examples across the development and validation sets. Do not use it. |

The existing whole-image blur check itself flags three fog/water examples labeled
usable here. The experiment does not change its threshold or claim to solve that
existing limitation. Three development photos with ambiguous visual degradation
were excluded from binary scoring; no new production finding category is proposed.

## What the prototypes actually measure

`evaluate.py` calls the current `_check_image` on file bytes, then performs the
experimental checks separately. It never invokes rejection or changes image data.

For visual features, it average-pools 1024-by-1024 RGB pixels to 256-by-256 and
measures an 8-by-8 grid of patch color standard deviations and grayscale Laplacian
variances. The revised regional rule compares the outer quarter-width strips below
the top quarter of the image. It requires low absolute detail and contrast on one
side and much stronger detail on the opposite side. Exact experimental thresholds
are in `regional_v2`; they are not approved production defaults.

The rule is deliberately narrow. It misses the controlled Gaussian-blur example
and does not cover arbitrary central blur, top/bottom blur, milder seams, or noise.
The raw-noise control also passes. These are explicit limits, not evidence that
such images are clean. The first regional rule and naive patch-count comparison
remain in the experiment so the evidence for choosing the revision is inspectable.

The near-blank prototype combines at least 85% low-contrast patches with dark,
bright or narrow-range luminance. It catches a controlled image with 87.5% black
area that the current checker misses, but the real tunnel counterexample prevents
recommending it as a general additional rejection check. The real matches are all
dark scenes; this run does not validate its bright-image branch.

The duplicate prototype hashes the exact decoded RGB buffer, prefixed with its
mode and dimensions. Comparison is limited to views within the same dataset and
location directory. It does not reject merely similar views or attempt to find
every recompressed copy: the progressive re-encoding control changes decoded
pixels and correctly has a different pixel hash.

The JPEG prototype uses libjpeg-turbo's full RGB decompression with
`TJFLAG_STOPONWARNING`, so decoder recovery warnings become failed checks.
The flag and API definitions were checked against the release's
[official header](https://raw.githubusercontent.com/libjpeg-turbo/libjpeg-turbo/3.2.0/src/turbojpeg.h).
This detects the tested malformed streams; no JPEG decoder can establish that a
validly encoded image contains the intended scene. CMYK/YCCK and images larger
than the experimental allocation limit are skipped by this standalone prototype.

## Runtime and eventual integration

For the 868 valid files, median measured times were:

| Operation | Median per image |
| --- | ---: |
| Current individual-image check | 37.59 ms |
| Additional strict JPEG decode | 2.76 ms |
| Decoded-pixel hash, with pixels already available | 2.02 ms |
| All exploratory visual features | 19.53 ms |

The original mixed 915-file run took 60.47 seconds. This is a local serial
prototype measurement, not a production throughput benchmark. The feature timing
includes both accepted and deferred experiments; it does not isolate regional v2.
Do not describe the visual checks as free or extrapolate a whole-dataset ETA from
this single run.

If implementation is authorized later, reuse already decoded pixels, restrict
these visual features to the expected dimensions/mode, and keep pixel hashes
scoped to each location. The shared image-reading path should support both normal
directories and tar members. A native decoder handle needs a defined lifetime per
worker; dependency availability and failures must not silently make every image
look corrupt. None of that integration or compatibility testing is claimed here.

## Reproduce and inspect

The experiment used Python 3.13.13, Pillow 12.1.0 and NumPy 2.4.2 from the existing
environment. No package dependency was changed. The official
[libjpeg-turbo 3.2.0 Windows x64 package](https://github.com/libjpeg-turbo/libjpeg-turbo/releases/tag/3.2.0)
was downloaded and extracted for its DLL; its installer was not run.

Package: `libjpeg-turbo-3.2.0-vc-x64.exe`

Verified package SHA-256:
`662761d8ba8dae04aec74023ebaeceb856c2b56b9b59cfd180759d26300dda42`

The extracted DLL is currently in the user's temporary directory. From the
repository root, with that file still present:

```powershell
$jpegDll = Join-Path $env:TEMP 'salty-jpeg-evaluation-3.2.0/portable/bin/turbojpeg.dll'
.\.venv\Scripts\python.exe -B experiments/image_quality/evaluate.py --turbojpeg $jpegDll --out experiments/image_quality/results
.\.venv\Scripts\python.exe -B experiments/image_quality/summarize.py
```

The script requires an explicitly supplied DLL and does not download one itself.
Dataset output paths are rejected. If the source dataset changes, the saved visual
audit may no longer cover the new predictions; the summary script then stops rather
than treating new predictions as visually confirmed. To inspect the existing
results without loading the decoder or rescanning photos, run only `summarize.py`.

- [metrics.json](results/metrics.json): original real-file measurements, input
  SHA-256 values, pixel hashes, feature grids, current-checker results and timings.
- [evaluation.json](results/evaluation.json): current decisions and scores,
  including regional v2 recomputed from the saved grids.
- [synthetic.json](results/synthetic.json): 13 controlled cases, including an
  unchanged copy and a distinct real heading as duplicate controls.
- [verification.json](results/verification.json): passing controlled assertions,
  panorama separation, and fresh verification of all 43 audited files. Their source
  hashes match the original run, and fresh feature decisions match cached decisions.
- [labels.json](labels.json), [validation_labels.json](validation_labels.json) and
  [candidate_labels.json](candidate_labels.json): individual visual judgments and notes.
- [holdout_selection.json](holdout_selection.json): validation selection recorded
  before its visual labels and prediction comparison.

`summary.json` and the real-file measurements preserve the first run, before
regional v2 was added. Use `evaluation.json` for the final comparison. Saved grids
are rounded to three decimals; fresh recomputation of the 43 audited candidates
confirmed that this rounding did not change their decisions.
