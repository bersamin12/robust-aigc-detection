# Robustness table

**Evaluation tier:** ablation

Selection tier: internal validation plus a stratified benchmark subsample, over the full grid. Used for ablations and model selection only. Planned budget for this tier: 5k internal validation plus a 5k stratified benchmark subsample, over the 20-condition grid. THIS TABLE covers 2 rung(s) x 20 condition(s) over 4000 image(s), which is what the numbers below were computed on; the budget above is the plan, not a property of this table.

**Bootstrap 95% CIs:** 1000 resamples, seed 20260827.

**§6.4 selection metric:** column `heldout_robust_tpr_at_1pct`, the rule the headline model is chosen by -- the rung among A3/A4/A5/A6 with the highest mean TPR @ 1% FPR over the degraded conditions, computed on val_internal authentic images vs heldout_generator generated images (spec §6.4). Fixed before any result existed; not clean AUC, not val_auc, not the external benchmark. Highest among the eligible rungs in this table: `a3`. The per-condition columns and `robust_auc` are §6.1 REPORTING metrics, computed over every scored row -- internal-validation authentic and generated, held-out-generator and benchmark alike. They are not the model-selection rule and can rank the rungs differently.

Conditions marked `(unseen)` use a severity the training sampler never drew (spec §4.6, held-out severity bands): jpeg_q70, blur_s1.0, social_repost, filtered_upload.

| rung | clean | jpeg_q90 | jpeg_q70 (unseen) | jpeg_q50 | jpeg_q30 | blur_s0.5 | blur_s1.0 (unseen) | blur_s2.0 | resize_0.5 | resize_0.25 | noise_s0.02 | noise_s0.05 | noise_s0.1 | jitter_20 | crop_80 | social_repost (unseen) | messaging_app | screenshot | filtered_upload (unseen) | low_light_share | robust_auc | heldout_auc | seen_auc | heldout_robust_tpr_at_1pct | tier | n_images | boot_seed | boot_n | banks_verified | target_fpr |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| a3 | 0.9743 | 0.9747 | 0.9730 | 0.9696 | 0.9610 | 0.9739 | 0.9716 | 0.9596 | 0.9732 | 0.9627 | 0.9639 | 0.9450 | 0.9141 | 0.9581 | 0.9618 | 0.9717 | 0.9464 | 0.9565 | 0.9571 | 0.9513 | 0.9603 | 0.9684 | 0.9581 | 0.6399 | ablation | 4000 | 20260827 | 1000 | True | 0.0100 |
| aF | 0.9734 | 0.9733 | 0.9726 | 0.9695 | 0.9606 | 0.9727 | 0.9708 | 0.9584 | 0.9724 | 0.9600 | 0.9641 | 0.9441 | 0.9121 | 0.9546 | 0.9621 | 0.9714 | 0.9453 | 0.9561 | 0.9533 | 0.9495 | 0.9591 | 0.9670 | 0.9570 | 0.6358 | ablation | 4000 | 20260827 | 1000 | True | 0.0100 |
