# Robustness table

**Evaluation tier:** ablation

Selection tier: internal validation plus a stratified benchmark subsample, over the full grid. Used for ablations and model selection only. Planned budget for this tier: 5k internal validation plus a 5k stratified benchmark subsample, over the 20-condition grid. THIS TABLE covers 5 rung(s) x 20 condition(s) over 4000 image(s), which is what the numbers below were computed on; the budget above is the plan, not a property of this table.

**Bootstrap 95% CIs:** 1000 resamples, seed 20260827.

**§6.4 selection metric:** column `heldout_robust_tpr_at_1pct`, the rule the headline model is chosen by -- the rung among A3/A4/A5/A6 with the highest mean TPR @ 1% FPR over the degraded conditions, computed on val_internal authentic images vs heldout_generator generated images (spec §6.4). Fixed before any result existed; not clean AUC, not val_auc, not the external benchmark. Highest among the eligible rungs in this table: `a3`. The per-condition columns and `robust_auc` are §6.1 REPORTING metrics, computed over every scored row -- internal-validation authentic and generated, held-out-generator and benchmark alike. They are not the model-selection rule and can rank the rungs differently.

Conditions marked `(unseen)` use a severity the training sampler never drew (spec §4.6, held-out severity bands): jpeg_q70, blur_s1.0, social_repost, filtered_upload.

| rung | clean | jpeg_q90 | jpeg_q70 (unseen) | jpeg_q50 | jpeg_q30 | blur_s0.5 | blur_s1.0 (unseen) | blur_s2.0 | resize_0.5 | resize_0.25 | noise_s0.02 | noise_s0.05 | noise_s0.1 | jitter_20 | crop_80 | social_repost (unseen) | messaging_app | screenshot | filtered_upload (unseen) | low_light_share | robust_auc | heldout_auc | seen_auc | heldout_robust_tpr_at_1pct | tier | n_images | boot_seed | boot_n | banks_verified | target_fpr |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| a0 | 0.9790 | 0.9786 | 0.9758 | 0.9725 | 0.9622 | 0.9786 | 0.9754 | 0.9595 | 0.9775 | 0.9666 | 0.9663 | 0.9378 | 0.8892 | 0.9561 | 0.9661 | 0.9739 | 0.9447 | 0.9576 | 0.9532 | 0.9428 | 0.9597 | 0.9696 | 0.9571 | 0.6829 | ablation | 4000 | 20260827 | 1000 | True | 0.0100 |
| a1 | 0.9766 | 0.9770 | 0.9756 | 0.9733 | 0.9645 | 0.9763 | 0.9745 | 0.9632 | 0.9759 | 0.9658 | 0.9666 | 0.9455 | 0.9141 | 0.9620 | 0.9649 | 0.9742 | 0.9494 | 0.9594 | 0.9601 | 0.9523 | 0.9629 | 0.9711 | 0.9607 | 0.6627 | ablation | 4000 | 20260827 | 1000 | True | 0.0100 |
| a2 | 0.9766 | 0.9770 | 0.9756 | 0.9733 | 0.9645 | 0.9763 | 0.9745 | 0.9632 | 0.9759 | 0.9658 | 0.9666 | 0.9455 | 0.9141 | 0.9620 | 0.9649 | 0.9742 | 0.9494 | 0.9594 | 0.9601 | 0.9523 | 0.9629 | 0.9711 | 0.9607 | 0.6627 | ablation | 4000 | 20260827 | 1000 | True | 0.0100 |
| a3 | 0.9731 | 0.9735 | 0.9722 | 0.9687 | 0.9601 | 0.9725 | 0.9703 | 0.9576 | 0.9720 | 0.9609 | 0.9614 | 0.9408 | 0.9094 | 0.9541 | 0.9605 | 0.9708 | 0.9434 | 0.9537 | 0.9532 | 0.9488 | 0.9581 | 0.9666 | 0.9558 | 0.6166 | ablation | 4000 | 20260827 | 1000 | True | 0.0100 |
| a7_norecon | 0.9697 | 0.9708 | 0.9707 | 0.9677 | 0.9580 | 0.9695 | 0.9676 | 0.9544 | 0.9687 | 0.9591 | 0.9612 | 0.9417 | 0.9113 | 0.9515 | 0.9573 | 0.9692 | 0.9438 | 0.9508 | 0.9513 | 0.9475 | 0.9564 | 0.9647 | 0.9542 | 0.6219 | ablation | 4000 | 20260827 | 1000 | True | 0.0100 |
