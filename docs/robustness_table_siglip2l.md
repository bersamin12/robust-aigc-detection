# Robustness table

**Evaluation tier:** ablation

Selection tier: internal validation plus a stratified benchmark subsample, over the full grid. Used for ablations and model selection only. Planned budget for this tier: 5k internal validation plus a 5k stratified benchmark subsample, over the 20-condition grid. THIS TABLE covers 6 rung(s) x 20 condition(s) over 25332 image(s), which is what the numbers below were computed on; the budget above is the plan, not a property of this table.

**Bootstrap 95% CIs:** 1000 resamples, seed 20260827.

**§6.4 selection metric:** column `heldout_robust_tpr_at_1pct`, the rule the headline model is chosen by -- the rung among A3/A4/A5/A6 with the highest mean TPR @ 1% FPR over the degraded conditions, computed on val_internal authentic images vs heldout_generator generated images (spec §6.4). Fixed before any result existed; not clean AUC, not val_auc, not the external benchmark. Highest among the eligible rungs in this table: `a5`. The per-condition columns and `robust_auc` are §6.1 REPORTING metrics, computed over every scored row -- internal-validation authentic and generated, held-out-generator and benchmark alike. They are not the model-selection rule and can rank the rungs differently.

Conditions marked `(unseen)` use a severity the training sampler never drew (spec §4.6, held-out severity bands): jpeg_q70, blur_s1.0, social_repost, filtered_upload.

| rung | clean | jpeg_q90 | jpeg_q70 (unseen) | jpeg_q50 | jpeg_q30 | blur_s0.5 | blur_s1.0 (unseen) | blur_s2.0 | resize_0.5 | resize_0.25 | noise_s0.02 | noise_s0.05 | noise_s0.1 | jitter_20 | crop_80 | social_repost (unseen) | messaging_app | screenshot | filtered_upload (unseen) | low_light_share | robust_auc | heldout_auc | seen_auc | heldout_robust_tpr_at_1pct | tier | n_images | boot_seed | boot_n | banks_verified | target_fpr |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| a0 | 0.9146 | 0.9022 | 0.8924 | 0.8817 | 0.8589 | 0.9191 | 0.9047 | 0.8096 | 0.9184 | 0.8653 | 0.7899 | 0.6663 | 0.6088 | 0.7871 | 0.9092 | 0.8954 | 0.7942 | 0.8592 | 0.7693 | 0.7348 | 0.8298 | 0.8655 | 0.8203 | 0.1870 | ablation | 25332 | 20260827 | 1000 | True | 0.0100 |
| a1 | 0.9225 | 0.9193 | 0.9162 | 0.9128 | 0.9041 | 0.9230 | 0.9147 | 0.8808 | 0.9231 | 0.8855 | 0.9000 | 0.8795 | 0.8589 | 0.8229 | 0.9133 | 0.9155 | 0.8499 | 0.8966 | 0.8200 | 0.8851 | 0.8906 | 0.8916 | 0.8903 | 0.2573 | ablation | 25332 | 20260827 | 1000 | True | 0.0100 |
| a2 | 0.9225 | 0.9193 | 0.9162 | 0.9128 | 0.9041 | 0.9230 | 0.9147 | 0.8808 | 0.9231 | 0.8855 | 0.9000 | 0.8795 | 0.8589 | 0.8229 | 0.9133 | 0.9155 | 0.8499 | 0.8966 | 0.8200 | 0.8851 | 0.8906 | 0.8916 | 0.8903 | 0.2573 | ablation | 25332 | 20260827 | 1000 | True | 0.0100 |
| a3 | 0.9323 | 0.9301 | 0.9269 | 0.9228 | 0.9135 | 0.9318 | 0.9246 | 0.8981 | 0.9321 | 0.8985 | 0.9205 | 0.8972 | 0.8755 | 0.8559 | 0.9234 | 0.9255 | 0.8721 | 0.9091 | 0.8501 | 0.9015 | 0.9058 | 0.9068 | 0.9055 | 0.2893 | ablation | 25332 | 20260827 | 1000 | True | 0.0100 |
| a7_norecon | 0.9333 | 0.9275 | 0.9207 | 0.9179 | 0.9125 | 0.9331 | 0.9230 | 0.8983 | 0.9323 | 0.9010 | 0.9074 | 0.8864 | 0.8716 | 0.8462 | 0.9327 | 0.9197 | 0.8722 | 0.9091 | 0.8369 | 0.8964 | 0.9024 | 0.9001 | 0.9030 | 0.2779 | ablation | 25332 | 20260827 | 1000 | True | 0.0100 |
| a5 | 0.9912 | 0.9916 | 0.9897 | 0.9882 | 0.9851 | 0.9912 | 0.9904 | 0.9867 | 0.9910 | 0.9843 | 0.9890 | 0.9829 | 0.9741 | 0.9819 | 0.9900 | 0.9895 | 0.9781 | 0.9860 | 0.9803 | 0.9829 | 0.9859 | 0.9875 | 0.9855 | 0.8773 | ablation | 25332 | 20260827 | 1000 | True | 0.0100 |
