# Robustness table

**Evaluation tier:** ablation

Selection tier: internal validation plus a stratified benchmark subsample, over the full grid. Used for ablations and model selection only. Planned budget for this tier: 5k internal validation plus a 5k stratified benchmark subsample, over the 20-condition grid. THIS TABLE covers 5 rung(s) x 20 condition(s) over 25332 image(s), which is what the numbers below were computed on; the budget above is the plan, not a property of this table.

**Bootstrap 95% CIs:** 1000 resamples, seed 20260827.

**§6.4 selection metric:** column `heldout_robust_tpr_at_1pct`, the rule the headline model is chosen by -- the rung among A3/A4/A5/A6 with the highest mean TPR @ 1% FPR over the degraded conditions, computed on val_internal authentic images vs heldout_generator generated images (spec §6.4). Fixed before any result existed; not clean AUC, not val_auc, not the external benchmark. Highest among the eligible rungs in this table: `a3`. The per-condition columns and `robust_acc_fixed` are §6.1 REPORTING metrics, computed over every scored row -- internal-validation authentic and generated, held-out-generator and benchmark alike. They are not the model-selection rule and can rank the rungs differently.

Conditions marked `(unseen)` use a severity the training sampler never drew (spec §4.6, held-out severity bands): jpeg_q70, blur_s1.0, social_repost, filtered_upload.

| rung | clean | jpeg_q90 | jpeg_q70 (unseen) | jpeg_q50 | jpeg_q30 | blur_s0.5 | blur_s1.0 (unseen) | blur_s2.0 | resize_0.5 | resize_0.25 | noise_s0.02 | noise_s0.05 | noise_s0.1 | jitter_20 | crop_80 | social_repost (unseen) | messaging_app | screenshot | filtered_upload (unseen) | low_light_share | robust_acc_fixed | heldout_acc_fixed | seen_acc_fixed | heldout_robust_tpr_at_1pct | tier | n_images | boot_seed | boot_n | banks_verified | target_fpr | clean_threshold |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| a0 | 0.9617 | 0.9681 | 0.9454 | 0.9250 | 0.8983 | 0.9636 | 0.9657 | 0.9673 | 0.9576 | 0.9398 | 0.9359 | 0.8926 | 0.8426 | 0.9552 | 0.9584 | 0.9452 | 0.8898 | 0.9235 | 0.9482 | 0.8790 | 0.9316 | 0.9511 | 0.9264 | 0.8611 | ablation | 25332 | 20260827 | 1000 | True | 0.0100 | 0.3812 |
| a1 | 0.9647 | 0.9714 | 0.9604 | 0.9536 | 0.9434 | 0.9664 | 0.9677 | 0.9670 | 0.9612 | 0.9462 | 0.9563 | 0.9454 | 0.9286 | 0.9602 | 0.9606 | 0.9612 | 0.9421 | 0.9490 | 0.9593 | 0.9444 | 0.9550 | 0.9622 | 0.9531 | 0.9037 | ablation | 25332 | 20260827 | 1000 | True | 0.0100 | 1.5645 |
| a2 | 0.9647 | 0.9714 | 0.9604 | 0.9536 | 0.9434 | 0.9664 | 0.9677 | 0.9670 | 0.9612 | 0.9462 | 0.9563 | 0.9454 | 0.9286 | 0.9602 | 0.9606 | 0.9612 | 0.9421 | 0.9490 | 0.9593 | 0.9444 | 0.9550 | 0.9622 | 0.9531 | 0.9037 | ablation | 25332 | 20260827 | 1000 | True | 0.0100 | 1.5645 |
| a3 | 0.9673 | 0.9713 | 0.9625 | 0.9554 | 0.9470 | 0.9686 | 0.9690 | 0.9669 | 0.9649 | 0.9520 | 0.9623 | 0.9528 | 0.9353 | 0.9634 | 0.9642 | 0.9626 | 0.9477 | 0.9539 | 0.9613 | 0.9480 | 0.9584 | 0.9638 | 0.9569 | 0.9012 | ablation | 25332 | 20260827 | 1000 | True | 0.0100 | -1.5532 |
| a7_norecon | 0.4301 | 0.4094 | 0.4058 | 0.4051 | 0.4015 | 0.4302 | 0.4291 | 0.4084 | 0.4274 | 0.4123 | 0.4114 | 0.4054 | 0.3915 | 0.4344 | 0.4085 | 0.4055 | 0.4013 | 0.3971 | 0.4050 | 0.3926 | 0.4096 | 0.4113 | 0.4091 | 0.0296 | ablation | 25332 | 20260827 | 1000 | True | 0.0100 | 0.1974 |
