# Robustness table

**Evaluation tier:** ablation

Selection tier: internal validation plus a stratified benchmark subsample, over the full grid. Used for ablations and model selection only. Planned budget for this tier: 5k internal validation plus a 5k stratified benchmark subsample, over the 20-condition grid. THIS TABLE covers 5 rung(s) x 20 condition(s) over 25332 image(s), which is what the numbers below were computed on; the budget above is the plan, not a property of this table.

**Bootstrap 95% CIs:** 1000 resamples, seed 20260827.

**§6.4 selection metric:** column `heldout_robust_tpr_at_1pct`, the rule the headline model is chosen by -- the rung among A3/A4/A5/A6 with the highest mean TPR @ 1% FPR over the degraded conditions, computed on val_internal authentic images vs heldout_generator generated images (spec §6.4). Fixed before any result existed; not clean AUC, not val_auc, not the external benchmark. Highest among the eligible rungs in this table: `a3`. The per-condition columns and `robust_auc` are §6.1 REPORTING metrics, computed over every scored row -- internal-validation authentic and generated, held-out-generator and benchmark alike. They are not the model-selection rule and can rank the rungs differently.

Conditions marked `(unseen)` use a severity the training sampler never drew (spec §4.6, held-out severity bands): jpeg_q70, blur_s1.0, social_repost, filtered_upload.

| rung | clean | jpeg_q90 | jpeg_q70 (unseen) | jpeg_q50 | jpeg_q30 | blur_s0.5 | blur_s1.0 (unseen) | blur_s2.0 | resize_0.5 | resize_0.25 | noise_s0.02 | noise_s0.05 | noise_s0.1 | jitter_20 | crop_80 | social_repost (unseen) | messaging_app | screenshot | filtered_upload (unseen) | low_light_share | robust_auc | heldout_auc | seen_auc | heldout_robust_tpr_at_1pct | tier | n_images | boot_seed | boot_n | banks_verified | target_fpr |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| a0 | 0.9935 | 0.9948 | 0.9901 | 0.9860 | 0.9794 | 0.9938 | 0.9941 | 0.9943 | 0.9926 | 0.9876 | 0.9858 | 0.9729 | 0.9553 | 0.9907 | 0.9925 | 0.9899 | 0.9747 | 0.9828 | 0.9891 | 0.9752 | 0.9854 | 0.9908 | 0.9839 | 0.8611 | ablation | 25332 | 20260827 | 1000 | True | 0.0100 |
| a1 | 0.9931 | 0.9943 | 0.9915 | 0.9900 | 0.9877 | 0.9934 | 0.9937 | 0.9930 | 0.9924 | 0.9881 | 0.9904 | 0.9870 | 0.9811 | 0.9915 | 0.9913 | 0.9915 | 0.9857 | 0.9877 | 0.9907 | 0.9875 | 0.9899 | 0.9919 | 0.9894 | 0.9037 | ablation | 25332 | 20260827 | 1000 | True | 0.0100 |
| a2 | 0.9931 | 0.9943 | 0.9915 | 0.9900 | 0.9877 | 0.9934 | 0.9937 | 0.9930 | 0.9924 | 0.9881 | 0.9904 | 0.9870 | 0.9811 | 0.9915 | 0.9913 | 0.9915 | 0.9857 | 0.9877 | 0.9907 | 0.9875 | 0.9899 | 0.9919 | 0.9894 | 0.9037 | ablation | 25332 | 20260827 | 1000 | True | 0.0100 |
| a3 | 0.9950 | 0.9954 | 0.9934 | 0.9919 | 0.9899 | 0.9950 | 0.9950 | 0.9939 | 0.9947 | 0.9919 | 0.9931 | 0.9898 | 0.9837 | 0.9936 | 0.9936 | 0.9933 | 0.9882 | 0.9905 | 0.9924 | 0.9891 | 0.9920 | 0.9935 | 0.9916 | 0.9012 | ablation | 25332 | 20260827 | 1000 | True | 0.0100 |
| a7_norecon | 0.5485 | 0.5330 | 0.5312 | 0.5303 | 0.5273 | 0.5485 | 0.5482 | 0.5342 | 0.5466 | 0.5363 | 0.5350 | 0.5304 | 0.5201 | 0.5514 | 0.5336 | 0.5308 | 0.5274 | 0.5253 | 0.5301 | 0.5220 | 0.5338 | 0.5351 | 0.5334 | 0.0296 | ablation | 25332 | 20260827 | 1000 | True | 0.0100 |
