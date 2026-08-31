# Robustness table

**Evaluation tier:** ablation

Selection tier: internal validation plus a stratified benchmark subsample, over the full grid. Used for ablations and model selection only. Planned budget for this tier: 5k internal validation plus a 5k stratified benchmark subsample, over the 20-condition grid. THIS TABLE covers 5 rung(s) x 20 condition(s) over 4000 image(s), which is what the numbers below were computed on; the budget above is the plan, not a property of this table.

**Bootstrap 95% CIs:** 1000 resamples, seed 20260827.

**§6.4 selection metric:** column `heldout_robust_tpr_at_1pct`, the rule the headline model is chosen by -- the rung among A3/A4/A5/A6 with the highest mean TPR @ 1% FPR over the degraded conditions, computed on val_internal authentic images vs heldout_generator generated images (spec §6.4). Fixed before any result existed; not clean AUC, not val_auc, not the external benchmark. Highest among the eligible rungs in this table: `a3`. The per-condition columns and `robust_auc` are §6.1 REPORTING metrics, computed over every scored row -- internal-validation authentic and generated, held-out-generator and benchmark alike. They are not the model-selection rule and can rank the rungs differently.

Conditions marked `(unseen)` use a severity the training sampler never drew (spec §4.6, held-out severity bands): jpeg_q70, blur_s1.0, social_repost, filtered_upload.

| rung | clean | jpeg_q90 | jpeg_q70 (unseen) | jpeg_q50 | jpeg_q30 | blur_s0.5 | blur_s1.0 (unseen) | blur_s2.0 | resize_0.5 | resize_0.25 | noise_s0.02 | noise_s0.05 | noise_s0.1 | jitter_20 | crop_80 | social_repost (unseen) | messaging_app | screenshot | filtered_upload (unseen) | low_light_share | robust_auc | heldout_auc | seen_auc | heldout_robust_tpr_at_1pct | tier | n_images | boot_seed | boot_n | banks_verified | target_fpr |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| a0 | 0.9670 | 0.9654 | 0.9632 | 0.9608 | 0.9553 | 0.9669 | 0.9671 | 0.9646 | 0.9662 | 0.9586 | 0.9563 | 0.9364 | 0.8941 | 0.9632 | 0.9589 | 0.9623 | 0.9446 | 0.9521 | 0.9592 | 0.9408 | 0.9545 | 0.9630 | 0.9523 | 0.6751 | ablation | 4000 | 20260827 | 1000 | True | 0.0100 |
| a1 | 0.9760 | 0.9745 | 0.9727 | 0.9703 | 0.9644 | 0.9761 | 0.9763 | 0.9743 | 0.9756 | 0.9682 | 0.9679 | 0.9541 | 0.9309 | 0.9713 | 0.9678 | 0.9719 | 0.9541 | 0.9617 | 0.9670 | 0.9556 | 0.9660 | 0.9720 | 0.9645 | 0.7356 | ablation | 4000 | 20260827 | 1000 | True | 0.0100 |
| a2 | 0.9760 | 0.9745 | 0.9727 | 0.9703 | 0.9644 | 0.9761 | 0.9763 | 0.9743 | 0.9756 | 0.9682 | 0.9679 | 0.9541 | 0.9309 | 0.9713 | 0.9678 | 0.9719 | 0.9541 | 0.9617 | 0.9670 | 0.9556 | 0.9660 | 0.9720 | 0.9645 | 0.7356 | ablation | 4000 | 20260827 | 1000 | True | 0.0100 |
| a3 | 0.9719 | 0.9707 | 0.9688 | 0.9662 | 0.9621 | 0.9718 | 0.9721 | 0.9697 | 0.9710 | 0.9649 | 0.9651 | 0.9534 | 0.9312 | 0.9679 | 0.9655 | 0.9680 | 0.9530 | 0.9595 | 0.9643 | 0.9537 | 0.9631 | 0.9683 | 0.9617 | 0.7355 | ablation | 4000 | 20260827 | 1000 | True | 0.0100 |
| a7_norecon | 0.9719 | 0.9705 | 0.9691 | 0.9670 | 0.9623 | 0.9719 | 0.9721 | 0.9697 | 0.9711 | 0.9647 | 0.9634 | 0.9509 | 0.9290 | 0.9674 | 0.9649 | 0.9681 | 0.9529 | 0.9598 | 0.9640 | 0.9532 | 0.9627 | 0.9683 | 0.9613 | 0.7091 | ablation | 4000 | 20260827 | 1000 | True | 0.0100 |
