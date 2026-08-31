# Robustness table

**Evaluation tier:** ablation

Selection tier: internal validation plus a stratified benchmark subsample, over the full grid. Used for ablations and model selection only. Planned budget for this tier: 5k internal validation plus a 5k stratified benchmark subsample, over the 20-condition grid. THIS TABLE covers 5 rung(s) x 20 condition(s) over 4000 image(s), which is what the numbers below were computed on; the budget above is the plan, not a property of this table.

**Bootstrap 95% CIs:** 1000 resamples, seed 20260827.

**§6.4 selection metric:** column `heldout_robust_tpr_at_1pct`, the rule the headline model is chosen by -- the rung among A3/A4/A5/A6 with the highest mean TPR @ 1% FPR over the degraded conditions, computed on val_internal authentic images vs heldout_generator generated images (spec §6.4). Fixed before any result existed; not clean AUC, not val_auc, not the external benchmark. Highest among the eligible rungs in this table: `a3`. The per-condition columns and `robust_auc` are §6.1 REPORTING metrics, computed over every scored row -- internal-validation authentic and generated, held-out-generator and benchmark alike. They are not the model-selection rule and can rank the rungs differently.

Conditions marked `(unseen)` use a severity the training sampler never drew (spec §4.6, held-out severity bands): jpeg_q70, blur_s1.0, social_repost, filtered_upload.

| rung | clean | jpeg_q90 | jpeg_q70 (unseen) | jpeg_q50 | jpeg_q30 | blur_s0.5 | blur_s1.0 (unseen) | blur_s2.0 | resize_0.5 | resize_0.25 | noise_s0.02 | noise_s0.05 | noise_s0.1 | jitter_20 | crop_80 | social_repost (unseen) | messaging_app | screenshot | filtered_upload (unseen) | low_light_share | robust_auc | heldout_auc | seen_auc | heldout_robust_tpr_at_1pct | tier | n_images | boot_seed | boot_n | banks_verified | target_fpr |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| a0 | 0.9450 | 0.9411 | 0.9323 | 0.9171 | 0.8961 | 0.9440 | 0.9385 | 0.9226 | 0.9421 | 0.9232 | 0.8895 | 0.8384 | 0.7981 | 0.9398 | 0.9374 | 0.9281 | 0.8874 | 0.9179 | 0.9234 | 0.8479 | 0.9087 | 0.9306 | 0.9028 | 0.5072 | ablation | 4000 | 20260827 | 1000 | True | 0.0100 |
| a1 | 0.9393 | 0.9379 | 0.9337 | 0.9255 | 0.9107 | 0.9390 | 0.9376 | 0.9300 | 0.9389 | 0.9248 | 0.9190 | 0.9014 | 0.8804 | 0.9355 | 0.9312 | 0.9326 | 0.9058 | 0.9216 | 0.9283 | 0.8975 | 0.9227 | 0.9331 | 0.9199 | 0.5055 | ablation | 4000 | 20260827 | 1000 | True | 0.0100 |
| a2 | 0.9393 | 0.9379 | 0.9337 | 0.9255 | 0.9107 | 0.9390 | 0.9376 | 0.9300 | 0.9389 | 0.9248 | 0.9190 | 0.9014 | 0.8804 | 0.9355 | 0.9312 | 0.9326 | 0.9058 | 0.9216 | 0.9283 | 0.8975 | 0.9227 | 0.9331 | 0.9199 | 0.5055 | ablation | 4000 | 20260827 | 1000 | True | 0.0100 |
| a3 | 0.9293 | 0.9290 | 0.9248 | 0.9164 | 0.9008 | 0.9282 | 0.9258 | 0.9140 | 0.9284 | 0.9123 | 0.9113 | 0.8913 | 0.8682 | 0.9230 | 0.9239 | 0.9236 | 0.8900 | 0.9122 | 0.9178 | 0.8850 | 0.9119 | 0.9230 | 0.9089 | 0.4036 | ablation | 4000 | 20260827 | 1000 | True | 0.0100 |
| a7_norecon | 0.9433 | 0.9421 | 0.9397 | 0.9323 | 0.9189 | 0.9432 | 0.9413 | 0.9300 | 0.9425 | 0.9302 | 0.9250 | 0.9043 | 0.8830 | 0.9375 | 0.9397 | 0.9374 | 0.9100 | 0.9272 | 0.9316 | 0.9029 | 0.9273 | 0.9375 | 0.9246 | 0.5482 | ablation | 4000 | 20260827 | 1000 | True | 0.0100 |
