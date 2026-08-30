# Robustness table

**Evaluation tier:** ablation

Selection tier: internal validation plus a stratified benchmark subsample, over the full grid. Used for ablations and model selection only. Planned budget for this tier: 5k internal validation plus a 5k stratified benchmark subsample, over the 20-condition grid. THIS TABLE covers 5 rung(s) x 20 condition(s) over 25332 image(s), which is what the numbers below were computed on; the budget above is the plan, not a property of this table.

**Bootstrap 95% CIs:** 1000 resamples, seed 20260827.

**§6.4 selection metric:** column `heldout_robust_tpr_at_1pct`, the rule the headline model is chosen by -- the rung among A3/A4/A5/A6 with the highest mean TPR @ 1% FPR over the degraded conditions, computed on val_internal authentic images vs heldout_generator generated images (spec §6.4). Fixed before any result existed; not clean AUC, not val_auc, not the external benchmark. Highest among the eligible rungs in this table: `a3`. The per-condition columns and `robust_acc_oracle` are §6.1 REPORTING metrics, computed over every scored row -- internal-validation authentic and generated, held-out-generator and benchmark alike. They are not the model-selection rule and can rank the rungs differently.

Conditions marked `(unseen)` use a severity the training sampler never drew (spec §4.6, held-out severity bands): jpeg_q70, blur_s1.0, social_repost, filtered_upload.

| rung | clean | jpeg_q90 | jpeg_q70 (unseen) | jpeg_q50 | jpeg_q30 | blur_s0.5 | blur_s1.0 (unseen) | blur_s2.0 | resize_0.5 | resize_0.25 | noise_s0.02 | noise_s0.05 | noise_s0.1 | jitter_20 | crop_80 | social_repost (unseen) | messaging_app | screenshot | filtered_upload (unseen) | low_light_share | robust_acc_oracle | heldout_acc_oracle | seen_acc_oracle | heldout_robust_tpr_at_1pct | tier | n_images | boot_seed | boot_n | banks_verified | target_fpr |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| a0 | 0.9647 | 0.9701 | 0.9547 | 0.9419 | 0.9252 | 0.9664 | 0.9678 | 0.9684 | 0.9610 | 0.9475 | 0.9445 | 0.9130 | 0.8809 | 0.9573 | 0.9604 | 0.9535 | 0.9197 | 0.9336 | 0.9527 | 0.9182 | 0.9440 | 0.9572 | 0.9405 | 0.8611 | ablation | 25332 | 20260827 | 1000 | True | 0.0100 |
| a1 | 0.9683 | 0.9731 | 0.9647 | 0.9600 | 0.9545 | 0.9691 | 0.9709 | 0.9687 | 0.9666 | 0.9541 | 0.9619 | 0.9535 | 0.9377 | 0.9624 | 0.9628 | 0.9648 | 0.9480 | 0.9539 | 0.9613 | 0.9527 | 0.9600 | 0.9654 | 0.9586 | 0.9037 | ablation | 25332 | 20260827 | 1000 | True | 0.0100 |
| a2 | 0.9683 | 0.9731 | 0.9647 | 0.9600 | 0.9545 | 0.9691 | 0.9709 | 0.9687 | 0.9666 | 0.9541 | 0.9619 | 0.9535 | 0.9377 | 0.9624 | 0.9628 | 0.9648 | 0.9480 | 0.9539 | 0.9613 | 0.9527 | 0.9600 | 0.9654 | 0.9586 | 0.9037 | ablation | 25332 | 20260827 | 1000 | True | 0.0100 |
| a3 | 0.9704 | 0.9734 | 0.9662 | 0.9614 | 0.9551 | 0.9708 | 0.9713 | 0.9681 | 0.9691 | 0.9589 | 0.9653 | 0.9569 | 0.9419 | 0.9661 | 0.9651 | 0.9660 | 0.9522 | 0.9557 | 0.9634 | 0.9535 | 0.9621 | 0.9667 | 0.9609 | 0.9012 | ablation | 25332 | 20260827 | 1000 | True | 0.0100 |
| a7_norecon | 0.6414 | 0.6410 | 0.6409 | 0.6406 | 0.6405 | 0.6414 | 0.6414 | 0.6418 | 0.6411 | 0.6414 | 0.6403 | 0.6401 | 0.6399 | 0.6416 | 0.6418 | 0.6407 | 0.6405 | 0.6409 | 0.6408 | 0.6402 | 0.6409 | 0.6409 | 0.6409 | 0.0296 | ablation | 25332 | 20260827 | 1000 | True | 0.0100 |
