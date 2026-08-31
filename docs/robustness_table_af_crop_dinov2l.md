# Robustness table

**Evaluation tier:** ablation

Selection tier: internal validation plus a stratified benchmark subsample, over the full grid. Used for ablations and model selection only. Planned budget for this tier: 5k internal validation plus a 5k stratified benchmark subsample, over the 20-condition grid. THIS TABLE covers 2 rung(s) x 20 condition(s) over 4000 image(s), which is what the numbers below were computed on; the budget above is the plan, not a property of this table.

**Bootstrap 95% CIs:** 1000 resamples, seed 20260827.

**§6.4 selection metric:** column `heldout_robust_tpr_at_1pct`, the rule the headline model is chosen by -- the rung among A3/A4/A5/A6 with the highest mean TPR @ 1% FPR over the degraded conditions, computed on val_internal authentic images vs heldout_generator generated images (spec §6.4). Fixed before any result existed; not clean AUC, not val_auc, not the external benchmark. Highest among the eligible rungs in this table: `a3`. The per-condition columns and `robust_auc` are §6.1 REPORTING metrics, computed over every scored row -- internal-validation authentic and generated, held-out-generator and benchmark alike. They are not the model-selection rule and can rank the rungs differently.

Conditions marked `(unseen)` use a severity the training sampler never drew (spec §4.6, held-out severity bands): jpeg_q70, blur_s1.0, social_repost, filtered_upload.

| rung | clean | jpeg_q90 | jpeg_q70 (unseen) | jpeg_q50 | jpeg_q30 | blur_s0.5 | blur_s1.0 (unseen) | blur_s2.0 | resize_0.5 | resize_0.25 | noise_s0.02 | noise_s0.05 | noise_s0.1 | jitter_20 | crop_80 | social_repost (unseen) | messaging_app | screenshot | filtered_upload (unseen) | low_light_share | robust_auc | heldout_auc | seen_auc | heldout_robust_tpr_at_1pct | tier | n_images | boot_seed | boot_n | banks_verified | target_fpr |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| a3 | 0.9701 | 0.9691 | 0.9671 | 0.9644 | 0.9598 | 0.9702 | 0.9705 | 0.9678 | 0.9693 | 0.9627 | 0.9635 | 0.9509 | 0.9259 | 0.9655 | 0.9623 | 0.9663 | 0.9500 | 0.9563 | 0.9620 | 0.9511 | 0.9608 | 0.9665 | 0.9592 | 0.7047 | ablation | 4000 | 20260827 | 1000 | True | 0.0100 |
| aF | 0.9748 | 0.9737 | 0.9721 | 0.9698 | 0.9651 | 0.9752 | 0.9754 | 0.9731 | 0.9744 | 0.9678 | 0.9681 | 0.9576 | 0.9388 | 0.9705 | 0.9688 | 0.9715 | 0.9550 | 0.9637 | 0.9670 | 0.9577 | 0.9666 | 0.9715 | 0.9653 | 0.7355 | ablation | 4000 | 20260827 | 1000 | True | 0.0100 |
