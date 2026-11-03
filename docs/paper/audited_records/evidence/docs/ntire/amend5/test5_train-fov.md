# Test 5: training field of view after augmentation

Amendment 5, Test 5. Branch `researcher/test5-train-fov`, from `bbb1306`
(amendment 6). Run 2026-10-19. CPU only, 98 s.

## 1. Summary & results

**Registered reading: the "narrow training band" sentence must be
rewritten.** The registered rule reads: "if the post-augmentation p5-p95
range is wider than Table 1's native range by more than 0.10 FOV at either
end, the 'narrow training band' sentence is rewritten with the true range."
The post-augmentation range over all 11 views is **0.391 to 1.000**, against
Table 1's 0.391 to 0.521. The high end extends by **+0.479**, almost five
times the 0.10 threshold. The low end extends by 0.000. The test decides no
claim.

| FOV of the 200 px window | n | p5 | p50 | p95 |
|---|---|---|---|---|
| train, Table 1 (native, all rows) | 407,223 | 0.391 | 0.391 | 0.521 |
| train, this sample, native | 20,000 | 0.391 | 0.391 | 0.521 |
| train, this sample, after augmentation (all 11 views) | 220,000 | 0.391 | 0.468 | 1.000 |
| train, after augmentation, trainer-weighted | 300,000* | 0.391 | 0.463 | 0.971 |

\* View 0 counted 5 times and each degraded view once, which matches the
trainer's 1/3 : 1/15 weighting (Section 4.3).

**Augmentation widens the window FOV upward only, at native pixel scale.**
NTIRE's permanent downscale (`downscale_keep`, in 25% of degraded views)
shrinks the image before the window is taken, so the window covers *more*
of the frame. No training window has FOV below 0.391, because no op
enlarges the image before the window. The eval images where the crop policy
loses most are the large ones, with FOV below 0.391 (70.8% of
`ntire_val_hard`). No native-scale training window covers them. The
wording in the paper is wrong: training is not close to a single value. A
quarter of all views (24.9%) fall above 0.521.

**Qualification (audit 2026-10-19, Section 5.6).** A second crop acts after
the window. The `crop` and `aspect_crop` ops, on 30.3% of views, crop the
n x n window and zoom back to n x n. Counting it, 18.0% of training views
show less than 200/512 of the frame (minimum 0.196), at higher
magnification. The registered definition measures the window before that
crop, so the registered reading above is unaffected. But "training never
shows a small fraction of the frame" is false. What training never shows is
a small fraction of the frame *at native pixel scale*.

## 2. Current state

Table 1 and Section 3 of `docs/paper/main.tex` describe the training FOV
from native sizes alone: "In training it always covers 39 to 52% of the
short side", and the caption says "Training is close to a single value".
The `ntire2` recipe pool, used to train A2 and B2swa, includes a
pre-canonical downscale. This changes the image size before the window is
taken, and Table 1 does not account for it.

## 3. Opportunities

- **Text fix.** A proposed rewrite is given in Section 6.3. It is not
  applied to `main.tex`.
- **Training-side test of the low end.** Training never sees FOV below
  0.391. An augmentation that enlarges the image before the window would
  cover the eval range below 0.391. Tests 7 and 8 are the registered
  training baselines. Neither targets this directly.

## 4. Testing methodology

### 4.1 Sample and settings

- Bank `data/banks/ntire2_meta`, the bank `scripts/ntire_v2.sh` trains A2
  (`ntire2_d24_518`) and B2swa (`ntire2_dual224`) on. Settings from the bank
  config: seed 20260827, recipe pool `ntire2`, max_ops 5, 11 views.
- Training flags from `ntire_v2.sh`: `--canon-mode crop --crop-side 200
  --crop-clamp`. Geometric augmentation is off and no family is excluded.
- Sample: 20,000 rows of `split == "train"` (407,223 rows), drawn with
  `default_rng(20261019).choice(..., replace=False)`.
- Image size: read from the file header of the exact file the trainer
  decodes (`manifest_root / rel_path`). These sizes equal the bank's
  `width` / `height` columns for all 20,000 sampled rows.

### 4.2 Following the trainer's pipeline

For each image and each view v = 0..10, the script follows
`features/extract.build_view`:

1. `recipe_for_view(v, row_id, seed, (), pool, max_ops)`, the trainer's own
   call. View 0 has the empty recipe.
2. `Recipe.split()`. Pre-canonical ops (`downscale_keep`) run through
   `apply_pre` on a zero array of the true shape. Only the output shape is
   used, and it does not depend on pixel values.
3. The window is `min(200, short side at that moment)` (`crop_clamp`),
   as in `canonical._random_square_crop`.

Logged per view: native short side; short side when the window is taken;
FOV = window / that short side; area coverage = window^2 / (h * w) at that
moment.

**Check against the bank:** every one of the 220,000 recomputed recipes
equals the recipe stored in `views.parquet` (0 mismatches). The views
analysed are the views the bank and the trainer use.

### 4.3 n = 512 and n = 224

The registration asks for both input sizes. The crop window reads only
`crop_side`, not `nominal_side`, and the recipes do not depend on n either.
So FOV and area coverage are identical at n = 512 (branch A) and n = 224
(branch B). One set of numbers serves both.

### 4.4 Weightings

- **All 11 views, equal weight:** the registered quantity.
- **Degraded views only (1-10):** for reference.
- **Trainer-weighted:** each training step uses view 0 once and `m_deg` = 2
  views drawn uniformly from 1..10 (`PairedSampler._draw_degraded_views`).
  View 0 therefore carries weight 1/3 and each degraded view 1/15.

## 5. Findings

### 5.1 The sample reproduces Table 1

The native FOV of the 20,000 sampled images has p5 / p50 / p95 = 0.391 /
0.391 / 0.521, identical to Table 1's train row to 3 dp. Native short
sides range from 220 to 512 px (p5 384, p50 512).

### 5.2 Downscale widens the band upward only

![FOV histogram](test5_train-fov_fig.pdf)

*Figure 1.* FOV of the 200 px window in training, native (per image)
against after augmentation (all 11 views), log density. The grey band is
Table 1's 0.391-0.521. File: `docs/ntire/amend5/test5_train-fov_fig.pdf`.

| | all 11 views | degraded views | views with `downscale_keep` |
|---|---|---|---|
| n | 220,000 | 200,000 | 50,214 |
| FOV < 0.391 | 0.0% | 0.0% | 0.0% |
| FOV > 0.521 | 24.9% | 27.0% | 94.5% |
| FOV >= 0.8 | 11.2% | 12.3% | 48.8% |
| FOV = 1.0 (clamped) | 6.3% | 6.9% | 27.4% |

`downscale_keep` (scale 0.3-0.8) appears in 22.8% of all views and 25.1% of
degraded views. The pool weights it 3:1 within the resize family. The
short side at the window falls as low as 76 px, and 6.1% of views fall
below 200 px, where the window becomes the whole short side. In views that
contain the op, FOV p5 / p50 / p95 = 0.518 / 0.791 / 1.000.

### 5.3 Area coverage

| area coverage (window area / image area) | n | p5 | p25 | p50 | p75 | p95 |
|---|---|---|---|---|---|---|
| native, per image | 20,000 | 0.096 | 0.114 | 0.149 | 0.153 | 0.222 |
| after augmentation, all 11 views | 220,000 | 0.100 | 0.122 | 0.150 | 0.218 | 0.660 |
| after augmentation, degraded views | 200,000 | 0.100 | 0.123 | 0.150 | 0.234 | 0.665 |
| after augmentation, trainer-weighted | 300,000 | 0.100 | 0.114 | 0.150 | 0.156 | 0.648 |

### 5.4 Post-window crops (not in the registered definition)

45.6% of views also contain a crop-family op (`crop`, `aspect_crop` or
`perspective`). These ops act on the n x n canonical window *after* it is
taken, and narrow the content further. They fall outside the registered
FOV definition ("window / short side at the moment the window is taken"),
so they are counted here and not folded into the numbers above.

### 5.6 Supplementary, not registered: FOV after the post-window crop

The training chain has two true crops (traced in code, audit 2026-10-19):

1. the 200 px window, taken from the decoded or `downscale_keep`-shrunk
   image (`canonical._random_square_crop`);
2. at most one crop-family op, applied after the window is resized to
   n x n. `crop` (`ops.center_crop`) keeps 0.75-0.98 of each side;
   `aspect_crop` (`ops.aspect_crop`) keeps 0.5-0.95 per axis. Both resize
   back to n x n, so the content is zoomed in. `perspective` is a +/-10%
   corner warp, not a crop, and is left at 1.0 here.

Visible fraction = registered FOV x the crop-#2 fraction. Two summaries:
the geometric mean over axes (`lin`), and the smaller axis (`min`).

| FOV (all 11 views) | p5 | p50 | p95 | min | share < 200/512 |
|---|---|---|---|---|---|
| registered (window moment) | 0.391 | 0.468 | 1.000 | 0.391 | 0.0% |
| after crop #2, `lin` | 0.296 | 0.415 | 0.942 | 0.196 | 18.0% |
| after crop #2, `min` | 0.265 | 0.404 | 0.942 | 0.195 | 18.9% |

Share of views per crop-family op: none 54.4%, `perspective` 15.3%,
`crop` 15.2%, `aspect_crop` 15.1%.

These small-FOV training views are zoomed in. A view showing 0.29 of the
short side via `crop` 0.75 presents 150 native px at n = 512 (x3.41). An
eval centre window at FOV 0.195 presents 200 native px (x2.56). In
training, a small fraction of the frame always comes with extra
magnification. At eval, it comes at fixed native scale. Which of the two
matters is the question Test 3 could not separate.

**Correction (2026-11-02, read-only code trace at `bbb1306`).** Branch A's
backbone resizes 512 to 518 (`backbones.py:595`), so the magnifications above
are x3.45 (not x3.41) and x2.59 (not x2.56). Table 6.1's native max 0.909 is
this 20,000-row sample. The full train set has two rows at short side 200, so
native FOV reaches 1.0. `perspective` was counted as FOV 1.0 but can zoom in
up to about 1.2x, and `aspect_crop` crops each side separately (stretch up to
1.9x). The floor is 0.5 x 200 / S = 0.195 at S = 512, as measured. No verdict
changes.

Chain-position note: training's downscale acts before the window, as
NTIRE's does. Training's crop-family ops act after it. NTIRE's
`randomcrop` / `randomaspectcrop` are, by assumption (our code does not
establish the organisers' order), applied to the full image before
delivery, so they are already reflected in the eval short side.

### 5.5 No label skew in the widened views

Views with FOV > 0.521 are 53.1% fake, against 51.7% over all views. The
widening does not differ materially between classes. This is expected,
since the recipe draw does not see the label.

## 6. Results

### 6.1 FOV, in Table 1's format plus quartiles

| FOV | n | p5 | p25 | p50 | p75 | p95 | min | max |
|---|---|---|---|---|---|---|---|---|
| native, per image | 20,000 | 0.391 | 0.391 | 0.391 | 0.481 | 0.521 | 0.391 | 0.909 |
| after augmentation, all 11 views | 220,000 | 0.391 | 0.391 | 0.468 | 0.521 | 1.000 | 0.391 | 1.000 |
| after augmentation, degraded views | 200,000 | 0.391 | 0.391 | 0.469 | 0.554 | 1.000 | 0.391 | 1.000 |
| after augmentation, trainer-weighted | 300,000 | 0.391 | 0.391 | 0.463 | 0.489 | 0.971 | 0.391 | 1.000 |

### 6.2 Registered reading

| | value |
|---|---|
| Table 1 native range (p5-p95) | 0.391-0.521 |
| post-augmentation range (all 11 views) | 0.391-1.000 |
| extension at the low end | 0.000 |
| extension at the high end | **+0.479** (threshold 0.10) |
| reading | **rewrite the sentence** |

### 6.3 Proposed rewrite (not applied)

`main.tex`, Section 3, currently: "In training it always covers 39 to 52%
of the short side."

Proposed: "Before augmentation, the window covers 39 to 52% of the short
side. Augmentation widens this in both directions. A pre-window downscale
shows the whole short side in 6% of views. A post-window crop shows as
little as 20% in 18% of views, at higher magnification. At native pixel
scale, no training view shows less than 39%."

The Table 1 caption currently says "Training is close to a single value."
Proposed: "Before augmentation, training is close to a single value." The
caption's "86.4% of images fall outside the training band of 0.391 to
0.521" can stay as a statement about native FOV (verified: 70.8% + 15.6%).
Alternatively: "70.8% fall below the smallest FOV of a native-scale
training window (200/512)". Verified: 1,770 of 2,500 rows have short side
> 512; with the threshold rounded to 0.391 the share is 72.7%. The earlier
draft of this section said "no training view covers less than 39%". That
is false once the post-window crop is counted (Section 5.6).

These are proposals for researcher and researcher. Test 12 collects text edits, and
this one belongs in its list.

## 7. Future work

- Add the Section 6.3 edit to Test 12's list.
- An augmentation that enlarges the image before the window, a pre-window
  upscale or random-resized crop at native scale, would cover FOV below
  0.391. Test 7 (multi-scale proxy) is the closest registered test.

## 8. What didn't work

- Nothing failed. Post-window crop ops are outside the registered
  definition (Section 5.4) and are reported separately.

## Files

- `scripts/amend5_test5.py`: the analysis. Run from the worktree root with
  `PYTHONPATH=$PWD/src`. Runtime 98 s.
- `docs/ntire/amend5/test5_train-fov.json`: every number above.
- `docs/ntire/amend5/test5_train-fov_fig.pdf`: Figure 1.
- `/scratch/anonymous/dion_extensions/outputs/test5/test5_train-fov_views.parquet`:
  per-view rows (220,000, 1.5 MB). Not in git, because the repo ignores
  `*.parquet`.

Numbers from commit `bbb1306ca202a9d179318def80cead0c66d1cc27`, with the
uncommitted script above.
