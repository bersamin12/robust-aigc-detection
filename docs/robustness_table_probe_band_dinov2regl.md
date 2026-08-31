# Robustness table

**Evaluation tier:** ablation

Selection tier: internal validation plus a stratified benchmark subsample, over the full grid. Used for ablations and model selection only. Planned budget for this tier: 5k internal validation plus a 5k stratified benchmark subsample, over the 20-condition grid. THIS TABLE covers 5 rung(s) x 20 condition(s) over 4000 image(s), which is what the numbers below were computed on; the budget above is the plan, not a property of this table.

**Bootstrap 95% CIs:** 1000 resamples, seed 20260827.

**§6.4 selection metric:** column `heldout_robust_tpr_at_1pct`, the rule the headline model is chosen by -- the rung among A3/A4/A5/A6 with the highest mean TPR @ 1% FPR over the degraded conditions, computed on val_internal authentic images vs heldout_generator generated images (spec §6.4). Fixed before any result existed; not clean AUC, not val_auc, not the external benchmark. Highest among the eligible rungs in this table: `a3`. The per-condition columns and `robust_auc` are §6.1 REPORTING metrics, computed over every scored row -- internal-validation authentic and generated, held-out-generator and benchmark alike. They are not the model-selection rule and can rank the rungs differently.

Conditions marked `(unseen)` use a severity the training sampler never drew (spec §4.6, held-out severity bands): jpeg_q70, blur_s1.0, social_repost, filtered_upload.

| rung | clean | jpeg_q90 | jpeg_q70 (unseen) | jpeg_q50 | jpeg_q30 | blur_s0.5 | blur_s1.0 (unseen) | blur_s2.0 | resize_0.5 | resize_0.25 | noise_s0.02 | noise_s0.05 | noise_s0.1 | jitter_20 | crop_80 | social_repost (unseen) | messaging_app | screenshot | filtered_upload (unseen) | low_light_share | robust_auc | heldout_auc | seen_auc | heldout_robust_tpr_at_1pct | tier | n_images | boot_seed | boot_n | banks_verified | target_fpr |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| a0 | 0.9770 | 0.9753 | 0.9731 | 0.9687 | 0.9627 | 0.9774 | 0.9774 | 0.9753 | 0.9762 | 0.9708 | 0.9623 | 0.9372 | 0.8961 | 0.9731 | 0.9713 | 0.9717 | 0.9568 | 0.9632 | 0.9702 | 0.9457 | 0.9634 | 0.9731 | 0.9608 | 0.7153 | ablation | 4000 | 20260827 | 1000 | True | 0.0100 |
| a1 | 0.9828 | 0.9821 | 0.9810 | 0.9791 | 0.9759 | 0.9829 | 0.9828 | 0.9820 | 0.9825 | 0.9797 | 0.9778 | 0.9698 | 0.9580 | 0.9809 | 0.9789 | 0.9803 | 0.9711 | 0.9748 | 0.9794 | 0.9719 | 0.9774 | 0.9808 | 0.9765 | 0.7844 | ablation | 4000 | 20260827 | 1000 | True | 0.0100 |
| a2 | 0.9828 | 0.9821 | 0.9810 | 0.9791 | 0.9759 | 0.9829 | 0.9828 | 0.9820 | 0.9825 | 0.9797 | 0.9778 | 0.9698 | 0.9580 | 0.9809 | 0.9789 | 0.9803 | 0.9711 | 0.9748 | 0.9794 | 0.9719 | 0.9774 | 0.9808 | 0.9765 | 0.7844 | ablation | 4000 | 20260827 | 1000 | True | 0.0100 |
| a3 | 0.9798 | 0.9785 | 0.9766 | 0.9746 | 0.9711 | 0.9796 | 0.9796 | 0.9784 | 0.9796 | 0.9758 | 0.9744 | 0.9668 | 0.9555 | 0.9770 | 0.9743 | 0.9760 | 0.9668 | 0.9691 | 0.9746 | 0.9693 | 0.9736 | 0.9767 | 0.9727 | 0.7242 | ablation | 4000 | 20260827 | 1000 | True | 0.0100 |
| a7_norecon | 0.9774 | 0.9763 | 0.9730 | 0.9700 | 0.9652 | 0.9773 | 0.9774 | 0.9764 | 0.9772 | 0.9728 | 0.9718 | 0.9632 | 0.9520 | 0.9738 | 0.9715 | 0.9717 | 0.9574 | 0.9631 | 0.9697 | 0.9633 | 0.9696 | 0.9730 | 0.9688 | 0.6698 | ablation | 4000 | 20260827 | 1000 | True | 0.0100 |
