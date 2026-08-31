# Robustness table

**Evaluation tier:** ablation

Selection tier: internal validation plus a stratified benchmark subsample, over the full grid. Used for ablations and model selection only. Planned budget for this tier: 5k internal validation plus a 5k stratified benchmark subsample, over the 20-condition grid. THIS TABLE covers 5 rung(s) x 20 condition(s) over 4000 image(s), which is what the numbers below were computed on; the budget above is the plan, not a property of this table.

**Bootstrap 95% CIs:** 1000 resamples, seed 20260827.

**§6.4 selection metric:** column `heldout_robust_tpr_at_1pct`, the rule the headline model is chosen by -- the rung among A3/A4/A5/A6 with the highest mean TPR @ 1% FPR over the degraded conditions, computed on val_internal authentic images vs heldout_generator generated images (spec §6.4). Fixed before any result existed; not clean AUC, not val_auc, not the external benchmark. Highest among the eligible rungs in this table: `a3`. The per-condition columns and `robust_auc` are §6.1 REPORTING metrics, computed over every scored row -- internal-validation authentic and generated, held-out-generator and benchmark alike. They are not the model-selection rule and can rank the rungs differently.

Conditions marked `(unseen)` use a severity the training sampler never drew (spec §4.6, held-out severity bands): jpeg_q70, blur_s1.0, social_repost, filtered_upload.

| rung | clean | jpeg_q90 | jpeg_q70 (unseen) | jpeg_q50 | jpeg_q30 | blur_s0.5 | blur_s1.0 (unseen) | blur_s2.0 | resize_0.5 | resize_0.25 | noise_s0.02 | noise_s0.05 | noise_s0.1 | jitter_20 | crop_80 | social_repost (unseen) | messaging_app | screenshot | filtered_upload (unseen) | low_light_share | robust_auc | heldout_auc | seen_auc | heldout_robust_tpr_at_1pct | tier | n_images | boot_seed | boot_n | banks_verified | target_fpr |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| a0 | 0.9690 | 0.9654 | 0.9573 | 0.9436 | 0.9277 | 0.9693 | 0.9635 | 0.9048 | 0.9669 | 0.9383 | 0.9058 | 0.8293 | 0.7958 | 0.9652 | 0.9634 | 0.9553 | 0.8816 | 0.9399 | 0.9563 | 0.8861 | 0.9271 | 0.9581 | 0.9189 | 0.4855 | ablation | 4000 | 20260827 | 1000 | True | 0.0100 |
| a1 | 0.9700 | 0.9691 | 0.9639 | 0.9598 | 0.9548 | 0.9705 | 0.9670 | 0.9528 | 0.9684 | 0.9529 | 0.9580 | 0.9443 | 0.9289 | 0.9666 | 0.9645 | 0.9631 | 0.9348 | 0.9542 | 0.9626 | 0.9468 | 0.9570 | 0.9642 | 0.9551 | 0.6172 | ablation | 4000 | 20260827 | 1000 | True | 0.0100 |
| a2 | 0.9700 | 0.9691 | 0.9639 | 0.9598 | 0.9548 | 0.9705 | 0.9670 | 0.9528 | 0.9684 | 0.9529 | 0.9580 | 0.9443 | 0.9289 | 0.9666 | 0.9645 | 0.9631 | 0.9348 | 0.9542 | 0.9626 | 0.9468 | 0.9570 | 0.9642 | 0.9551 | 0.6172 | ablation | 4000 | 20260827 | 1000 | True | 0.0100 |
| a3 | 0.9477 | 0.9468 | 0.9398 | 0.9319 | 0.9203 | 0.9468 | 0.9410 | 0.9149 | 0.9448 | 0.9231 | 0.9362 | 0.9169 | 0.8964 | 0.9416 | 0.9378 | 0.9379 | 0.8856 | 0.9209 | 0.9342 | 0.9116 | 0.9278 | 0.9382 | 0.9250 | 0.5138 | ablation | 4000 | 20260827 | 1000 | True | 0.0100 |
| a7_norecon | 0.9570 | 0.9560 | 0.9512 | 0.9472 | 0.9426 | 0.9582 | 0.9559 | 0.9458 | 0.9570 | 0.9458 | 0.9497 | 0.9396 | 0.9255 | 0.9528 | 0.9539 | 0.9526 | 0.9296 | 0.9453 | 0.9487 | 0.9363 | 0.9470 | 0.9521 | 0.9457 | 0.5824 | ablation | 4000 | 20260827 | 1000 | True | 0.0100 |
