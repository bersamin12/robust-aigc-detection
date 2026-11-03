# Claim–evidence map

All paths below are original repository paths. Anonymized JSON/Markdown snapshots are under `evidence/` with the same path hierarchy. `evidence_ledger.json` binds each to its original and packaged SHA-256. Scientific claims use existing reports only. Primary literature is separately documented in `literature_verification.md`.

| Claim / location | Recorded evidence and population | Status / limit |
|---|---|---|
| Empirical contribution, title/abstract | Frozen presentation probes, Tests 14/16/17; earlier resize literature | Supported as empirical analysis; resize itself is established. |
| Hard-val frozen fusion 0.8856→0.9631 | `docs/ntire/fusion_center_valfit.json`, `fusion_squish_valfit.json` | Supported; selection split, separately fitted presentation statistics. |
| SocialRF A 0.867656→0.958200, B 0.882892→0.962304 | `docs/ntire/amend5/test16_tile-factorial-socialrf.json`, branches.*.pooled.robust_center/robust_squish | Supported; 1,000 degraded rows, short side ≥800, 500 per label. +9.0544/+7.9412 points are derived point estimates, rounded +9.1/+7.9. No paired interval for these specific differences. |
| Initial single crop→native central-quarter tiling gains 5.34/5.10 points | Same source, robust_tiles_k1_f0.25 minus robust_center | Supported point estimates; 5.3364/5.0968 points. No saved paired interval. |
| Little additional coverage gain on degraded SocialRF at 25–100% area | Test 16 robust_dF and per-cell AUCs; Test 20 CLIP; Test 21 B-Free | Supported within this coverage range, scale construction, and benchmark. Does not imply coverage never helps. |
| Downscale benefits at fixed coverage | Tests 14/16 robust_dK | Supported; interpolation and number of windows also change. |
| DINOv2 round trips small, positive residual | `test17_roundtrip.json`, row_sets.*.*.pooled.robust_dA/robust_dS | Supported numerical contrasts; context interpretation remains tentative. Source dS is named dV in paper. |
| CLIP round trip also helps clean data | `test20_clip-decomp.json`, per-seed clean/robust dA | Supported; not evidence of degradation-specific removal. |
| Ordering grows from 2× to 4× | `test18_shrinkorder.json`, reduction_k2/k4/kmax_minus_k2 | Supported on 2,833 originally clean NTIRE sources. Average of operation-specific AUC contrasts, distinct from the factorial population. |
| 4× factorial evaluated only at full coverage | Tests 14/16 saved cells; `scripts/amend5_test14.py` | Supported; not a full 3×3 factorial. |
| In-band frozen gains +11.1 [5.9,17.0], +7.9 [3.6,12.9] points | `test12_writing.json`, bin_416_496.branches.*[squish at test] | Supported; 171 distorted hard-val rows (86 real/85 fake). Rules out native training-size mismatch as sole explanation. |
| Size-decile control | `docs/ntire/crop_probe.md` | Supported for B only; A result and aggregate CI unavailable. |
| Native training sizes 200–512 and field-of-view distribution | `docs/ntire/paper/external_sets.json`, `docs/ntire/amend5/test5_train-fov.json`, `docs/ntire/crop_probe.md` | Supported in supplied reports; 20,000 images/220,000 candidate views for augmentation analysis. |
| Historical 200-pixel rationale | `docs/resolution_shortcut.md`; `src/aigcdet/augment/canonical.py` | Earlier benchmark real-image information ceiling and minimum training side; not proof 200 is optimal. |
| A 200→512 canonicalization versus tower 518 | canonical.py; features/backbones.py; `docs/ntire/README.md` | Supported, 2.56× vs 2.59×; native-512 still resizes 512→518. |
| Candidate recipes versus actual training draws | `docs/ntire/README.md`; `scripts/ntire_v2.sh`, `paper_pipeline.sh`; models/sampler.py | Ten candidates, two degraded draws per source, 64 source draws/batch, replacement sampling, proportional family weights. Full epoch formula in supplement. |
| Fusion crop/frozen-whole/trained-whole/historical weights | `paper/fusion_crop.json`; `fusion_squish_valfit.json`; `paper/fusion_squish.json`; `fusion_ntire2_final.json` | Supported, 0.50/0.50/0.55 on val, historical 0.45 on val+hard. Mean/SD estimation separate from weight selection. |
| Full trained robust gain +0.074 [0.058,0.091] | `paper/paper_results.json`, tables.G2; branch_A/B | Supported on selection split; distinct models and fusion fit. |
| Trained-system size gradient | `paper/analysis_figsize.json` | Supported, eight equal-count distorted hard-val bins; moved to Training. |
| Three-seed proxy/CLIP training gains | `paper/squish_proxy_gate.md`, `clip_proxy_gate.md`, `clip_testtime.md`; Tests 7/8 | Supported; weaker/shorter proxies, not identical full branches. |
| Native-224 gap persists | `test8_native-crop-proxy.json` | Supported for three seeds; some native-224 vs enlarged-200 intervals include zero. |
| Native-512 single-seed all-distorted gain +0.0415 | `test15_native512-proxy.json` | Supported; all 1,250 distorted hard-val rows mix clamped and native crops. |
| Matched 512/196 comparison | `test19_stretch.json`, splits.ntire_val_hard.short_ge_512.robust | Supported on exactly 630 distorted images, 320 real/310 fake, no clamping. E +0.0115 [-0.0139,0.0365], V +0.0478 [0.0217,0.0725], T +0.0593 [0.0323,0.0874]. One seed, token count/augmentation/2% crop-size confounds. |
| Native-control clean losses | Same, short_ge_512.clean | Supported on a different 1,188 clean rows. No causal spectral/generator explanation established. |
| MIL improves grid9 but whole-image still wins | `test9_mil-combiner.json`, robust_squish_minus_combiner | Supported, validation-fitted combiner, frozen towers; does not rule out every possible aggregator. |
| External distributed/JPEG95/degraded AUCs and CIs | `paper/paper_results.json`, tables.ext_*; `fusion_crop.json`, `fusion_squish.json` | Supported; paired reports and standalone point differences agree to numerical precision. Pooled AUC is not a macro generator average. |
| Strict operating-point trade-off | `test12_writing.json`, tables; `test12_writing.md` §5.3 | Supported point estimates, no TPR CIs. Retrospective thresholds on each evaluated real population. Fused diagnosis and test TPRs are pooled. |
| Complete-system test gain and clean cost | `paper/read2_vs_read1.json/.md`; `fusion_ntire2_final.json`; `paper/fusion_squish_final.json` | Supported from saved reports, not reevaluated. 2,500 images, 50% distorted. Different checkpoints and fusion settings. |
| Freeze/evaluation dates | `docs/ntire/results.md`; registration; test-read scripts | Reported freeze September 23 vs read October 4; read 2 October 11 after fit. Independent exact read-2 freeze timestamp missing. |
| Clean loss localized above 1100 | `paper/analysis_testsize.md`; Test 11 | Supported exploratory localization, not prospective test finding. Later declared subgroup preserved as such. Cause unexplained. |
| B-Free SocialRF gain and CommunityAI/Synthbuster losses | `paper/public_detectors.json`, bfree.splits.ext_*_deg.boot.vs_baseline.squish.auc | Supported: +6.332 [5.130,7.502], −14.478 [−15.273,−13.682], −0.412 [−1.287,0.481] points. Synthbuster interval includes zero. |
| Large B-Free SocialRF tiled gain | `test21_bfree-tiles.json` | Supported on short side ≥2016, 1,544 total rows/772 degraded, different from DINOv2 population. |
| Format/size confounds | `paper/external_sets.json`; official original datasets | Supported for evaluated copies, not asserted for every version of datasets. |
| Two completed test reads; proposed third canceled | registration amendments 5/11; saved results; Test 6 gate failure | Reported history; no third read performed for this audit. |
| Historical preregistration of late amendments | Git first appearance 745fdc7; text dates in amendments 7–12 | Unverifiable pre-run ordering; newly anonymized packaging is not independent evidence. |
| Cause of cross-pass numerical drift | Probe vs as-trained JSONs | Unverifiable; max AUC difference 0.00009472, already correctly bounded by 0.0001. |
