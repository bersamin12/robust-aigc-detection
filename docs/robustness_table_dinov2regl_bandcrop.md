# Robustness table

**Evaluation tier:** ablation

Selection tier: internal validation plus a stratified benchmark subsample, over the full grid. Used for ablations and model selection only. Planned budget for this tier: 5k internal validation plus a 5k stratified benchmark subsample, over the 20-condition grid. THIS TABLE covers 2 rung(s) x 20 condition(s) over 4000 image(s), which is what the numbers below were computed on; the budget above is the plan, not a property of this table.

**Bootstrap 95% CIs:** 1000 resamples, seed 20260827.

**§6.4 selection metric:** column `heldout_robust_tpr_at_1pct`, the rule the headline model is chosen by -- the rung among A3/A4/A5/A6 with the highest mean TPR @ 1% FPR over the degraded conditions, computed on val_internal authentic images vs heldout_generator generated images (spec §6.4). Fixed before any result existed; not clean AUC, not val_auc, not the external benchmark. Highest among the eligible rungs in this table: `a5`. The per-condition columns and `robust_auc` are §6.1 REPORTING metrics, computed over every scored row -- internal-validation authentic and generated, held-out-generator and benchmark alike. They are not the model-selection rule and can rank the rungs differently.

Conditions marked `(unseen)` use a severity the training sampler never drew (spec §4.6, held-out severity bands): jpeg_q70, blur_s1.0, social_repost, filtered_upload.

| rung | clean | jpeg_q90 | jpeg_q70 (unseen) | jpeg_q50 | jpeg_q30 | blur_s0.5 | blur_s1.0 (unseen) | blur_s2.0 | resize_0.5 | resize_0.25 | noise_s0.02 | noise_s0.05 | noise_s0.1 | jitter_20 | crop_80 | social_repost (unseen) | messaging_app | screenshot | filtered_upload (unseen) | low_light_share | robust_auc | heldout_auc | seen_auc | heldout_robust_tpr_at_1pct | tier | n_images | boot_seed | boot_n | banks_verified | target_fpr |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| a3 | 0.9798 | 0.9785 | 0.9766 | 0.9746 | 0.9711 | 0.9796 | 0.9796 | 0.9784 | 0.9796 | 0.9758 | 0.9744 | 0.9668 | 0.9555 | 0.9770 | 0.9743 | 0.9760 | 0.9668 | 0.9691 | 0.9746 | 0.9693 | 0.9736 | 0.9767 | 0.9727 | 0.7242 | ablation | 4000 | 20260827 | 1000 | True | 0.0100 |
| a5 | 0.9890 | 0.9884 | 0.9873 | 0.9860 | 0.9834 | 0.9890 | 0.9890 | 0.9878 | 0.9887 | 0.9856 | 0.9856 | 0.9795 | 0.9687 | 0.9870 | 0.9850 | 0.9867 | 0.9790 | 0.9820 | 0.9858 | 0.9810 | 0.9845 | 0.9872 | 0.9838 | 0.8714 | ablation | 4000 | 20260827 | 1000 | True | 0.0100 |
