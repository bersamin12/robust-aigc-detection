# Robustness table

**Evaluation tier:** ablation

Selection tier: internal validation plus a stratified benchmark subsample, over the full grid. Used for ablations and model selection only. Planned budget for this tier: 5k internal validation plus a 5k stratified benchmark subsample, over the 20-condition grid. THIS TABLE covers 5 rung(s) x 20 condition(s) over 4000 image(s), which is what the numbers below were computed on; the budget above is the plan, not a property of this table.

**Bootstrap 95% CIs:** 1000 resamples, seed 20260827.

**§6.4 selection metric:** column `heldout_robust_tpr_at_1pct`, the rule the headline model is chosen by -- the rung among A3/A4/A5/A6 with the highest mean TPR @ 1% FPR over the degraded conditions, computed on val_internal authentic images vs heldout_generator generated images (spec §6.4). Fixed before any result existed; not clean AUC, not val_auc, not the external benchmark. Highest among the eligible rungs in this table: `a3`. The per-condition columns and `robust_auc` are §6.1 REPORTING metrics, computed over every scored row -- internal-validation authentic and generated, held-out-generator and benchmark alike. They are not the model-selection rule and can rank the rungs differently.

Conditions marked `(unseen)` use a severity the training sampler never drew (spec §4.6, held-out severity bands): jpeg_q70, blur_s1.0, social_repost, filtered_upload.

| rung | clean | jpeg_q90 | jpeg_q70 (unseen) | jpeg_q50 | jpeg_q30 | blur_s0.5 | blur_s1.0 (unseen) | blur_s2.0 | resize_0.5 | resize_0.25 | noise_s0.02 | noise_s0.05 | noise_s0.1 | jitter_20 | crop_80 | social_repost (unseen) | messaging_app | screenshot | filtered_upload (unseen) | low_light_share | robust_auc | heldout_auc | seen_auc | heldout_robust_tpr_at_1pct | tier | n_images | boot_seed | boot_n | banks_verified | target_fpr |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| a0 | 0.9908 | 0.9921 | 0.9855 | 0.9795 | 0.9712 | 0.9910 | 0.9911 | 0.9900 | 0.9895 | 0.9785 | 0.9836 | 0.9708 | 0.9496 | 0.9881 | 0.9878 | 0.9845 | 0.9616 | 0.9764 | 0.9862 | 0.9661 | 0.9802 | 0.9868 | 0.9784 | 0.8379 | ablation | 4000 | 20260827 | 1000 | True | 0.0100 |
| a1 | 0.9911 | 0.9924 | 0.9868 | 0.9829 | 0.9787 | 0.9912 | 0.9912 | 0.9895 | 0.9900 | 0.9811 | 0.9873 | 0.9807 | 0.9685 | 0.9897 | 0.9888 | 0.9860 | 0.9706 | 0.9804 | 0.9879 | 0.9756 | 0.9842 | 0.9880 | 0.9831 | 0.8248 | ablation | 4000 | 20260827 | 1000 | True | 0.0100 |
| a2 | 0.9911 | 0.9924 | 0.9868 | 0.9829 | 0.9787 | 0.9912 | 0.9912 | 0.9895 | 0.9900 | 0.9811 | 0.9873 | 0.9807 | 0.9685 | 0.9897 | 0.9888 | 0.9860 | 0.9706 | 0.9804 | 0.9879 | 0.9756 | 0.9842 | 0.9880 | 0.9831 | 0.8248 | ablation | 4000 | 20260827 | 1000 | True | 0.0100 |
| a3 | 0.9916 | 0.9930 | 0.9880 | 0.9848 | 0.9813 | 0.9917 | 0.9919 | 0.9905 | 0.9907 | 0.9834 | 0.9880 | 0.9826 | 0.9732 | 0.9901 | 0.9891 | 0.9873 | 0.9768 | 0.9819 | 0.9888 | 0.9798 | 0.9859 | 0.9890 | 0.9851 | 0.8667 | ablation | 4000 | 20260827 | 1000 | True | 0.0100 |
| a7_norecon | 0.9885 | 0.9910 | 0.9847 | 0.9813 | 0.9760 | 0.9885 | 0.9883 | 0.9868 | 0.9870 | 0.9774 | 0.9844 | 0.9771 | 0.9629 | 0.9870 | 0.9866 | 0.9839 | 0.9697 | 0.9779 | 0.9859 | 0.9733 | 0.9816 | 0.9857 | 0.9805 | 0.8237 | ablation | 4000 | 20260827 | 1000 | True | 0.0100 |
