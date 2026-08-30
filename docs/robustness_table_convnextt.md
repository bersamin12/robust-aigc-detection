# Robustness table

**Evaluation tier:** ablation

Selection tier: internal validation plus a stratified benchmark subsample, over the full grid. Used for ablations and model selection only. Planned budget for this tier: 5k internal validation plus a 5k stratified benchmark subsample, over the 20-condition grid. THIS TABLE covers 6 rung(s) x 20 condition(s) over 25332 image(s), which is what the numbers below were computed on; the budget above is the plan, not a property of this table.

**Bootstrap 95% CIs:** 1000 resamples, seed 20260827.

**§6.4 selection metric:** column `heldout_robust_tpr_at_1pct`, the rule the headline model is chosen by -- the rung among A3/A4/A5/A6 with the highest mean TPR @ 1% FPR over the degraded conditions, computed on val_internal authentic images vs heldout_generator generated images (spec §6.4). Fixed before any result existed; not clean AUC, not val_auc, not the external benchmark. Highest among the eligible rungs in this table: `a5`. The per-condition columns and `robust_auc` are §6.1 REPORTING metrics, computed over every scored row -- internal-validation authentic and generated, held-out-generator and benchmark alike. They are not the model-selection rule and can rank the rungs differently.

Conditions marked `(unseen)` use a severity the training sampler never drew (spec §4.6, held-out severity bands): jpeg_q70, blur_s1.0, social_repost, filtered_upload.

| rung | clean | jpeg_q90 | jpeg_q70 (unseen) | jpeg_q50 | jpeg_q30 | blur_s0.5 | blur_s1.0 (unseen) | blur_s2.0 | resize_0.5 | resize_0.25 | noise_s0.02 | noise_s0.05 | noise_s0.1 | jitter_20 | crop_80 | social_repost (unseen) | messaging_app | screenshot | filtered_upload (unseen) | low_light_share | robust_auc | heldout_auc | seen_auc | heldout_robust_tpr_at_1pct | tier | n_images | boot_seed | boot_n | banks_verified | target_fpr |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| a0 | 0.9527 | 0.9350 | 0.9234 | 0.8971 | 0.8688 | 0.9576 | 0.9410 | 0.8584 | 0.9579 | 0.8873 | 0.7590 | 0.7178 | 0.7069 | 0.9283 | 0.9528 | 0.9235 | 0.7973 | 0.8868 | 0.9079 | 0.7613 | 0.8720 | 0.9240 | 0.8582 | 0.4244 | ablation | 25332 | 20260827 | 1000 | True | 0.0100 |
| a1 | 0.9467 | 0.9456 | 0.9469 | 0.9436 | 0.9395 | 0.9479 | 0.9408 | 0.9212 | 0.9520 | 0.9384 | 0.9238 | 0.9217 | 0.9075 | 0.9354 | 0.9575 | 0.9475 | 0.9055 | 0.9443 | 0.9340 | 0.9380 | 0.9364 | 0.9423 | 0.9348 | 0.4967 | ablation | 25332 | 20260827 | 1000 | True | 0.0100 |
| a2 | 0.9467 | 0.9456 | 0.9469 | 0.9436 | 0.9395 | 0.9479 | 0.9408 | 0.9212 | 0.9520 | 0.9384 | 0.9238 | 0.9217 | 0.9075 | 0.9354 | 0.9575 | 0.9475 | 0.9055 | 0.9443 | 0.9340 | 0.9380 | 0.9364 | 0.9423 | 0.9348 | 0.4967 | ablation | 25332 | 20260827 | 1000 | True | 0.0100 |
| a3 | 0.9519 | 0.9502 | 0.9500 | 0.9479 | 0.9444 | 0.9520 | 0.9464 | 0.9244 | 0.9537 | 0.9425 | 0.9387 | 0.9352 | 0.9233 | 0.9462 | 0.9510 | 0.9505 | 0.9241 | 0.9457 | 0.9429 | 0.9421 | 0.9427 | 0.9474 | 0.9414 | 0.4882 | ablation | 25332 | 20260827 | 1000 | True | 0.0100 |
| a7_norecon | 0.9629 | 0.9609 | 0.9600 | 0.9577 | 0.9523 | 0.9611 | 0.9512 | 0.9342 | 0.9638 | 0.9501 | 0.9527 | 0.9504 | 0.9368 | 0.9475 | 0.9681 | 0.9570 | 0.9144 | 0.9534 | 0.9441 | 0.9445 | 0.9505 | 0.9531 | 0.9499 | 0.5589 | ablation | 25332 | 20260827 | 1000 | True | 0.0100 |
| a5 | 0.9931 | 0.9934 | 0.9914 | 0.9900 | 0.9883 | 0.9932 | 0.9925 | 0.9893 | 0.9930 | 0.9893 | 0.9899 | 0.9868 | 0.9808 | 0.9915 | 0.9931 | 0.9915 | 0.9840 | 0.9898 | 0.9899 | 0.9877 | 0.9898 | 0.9913 | 0.9894 | 0.8860 | ablation | 25332 | 20260827 | 1000 | True | 0.0100 |
