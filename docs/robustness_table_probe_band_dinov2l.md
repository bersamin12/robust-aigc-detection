# Robustness table

**Evaluation tier:** ablation

Selection tier: internal validation plus a stratified benchmark subsample, over the full grid. Used for ablations and model selection only. Planned budget for this tier: 5k internal validation plus a 5k stratified benchmark subsample, over the 20-condition grid. THIS TABLE covers 5 rung(s) x 20 condition(s) over 4000 image(s), which is what the numbers below were computed on; the budget above is the plan, not a property of this table.

**Bootstrap 95% CIs:** 1000 resamples, seed 20260827.

**§6.4 selection metric:** column `heldout_robust_tpr_at_1pct`, the rule the headline model is chosen by -- the rung among A3/A4/A5/A6 with the highest mean TPR @ 1% FPR over the degraded conditions, computed on val_internal authentic images vs heldout_generator generated images (spec §6.4). Fixed before any result existed; not clean AUC, not val_auc, not the external benchmark. Highest among the eligible rungs in this table: `a3`. The per-condition columns and `robust_auc` are §6.1 REPORTING metrics, computed over every scored row -- internal-validation authentic and generated, held-out-generator and benchmark alike. They are not the model-selection rule and can rank the rungs differently.

Conditions marked `(unseen)` use a severity the training sampler never drew (spec §4.6, held-out severity bands): jpeg_q70, blur_s1.0, social_repost, filtered_upload.

| rung | clean | jpeg_q90 | jpeg_q70 (unseen) | jpeg_q50 | jpeg_q30 | blur_s0.5 | blur_s1.0 (unseen) | blur_s2.0 | resize_0.5 | resize_0.25 | noise_s0.02 | noise_s0.05 | noise_s0.1 | jitter_20 | crop_80 | social_repost (unseen) | messaging_app | screenshot | filtered_upload (unseen) | low_light_share | robust_auc | heldout_auc | seen_auc | heldout_robust_tpr_at_1pct | tier | n_images | boot_seed | boot_n | banks_verified | target_fpr |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| a0 | 0.9745 | 0.9724 | 0.9693 | 0.9666 | 0.9600 | 0.9746 | 0.9741 | 0.9724 | 0.9742 | 0.9682 | 0.9612 | 0.9390 | 0.8974 | 0.9715 | 0.9674 | 0.9690 | 0.9558 | 0.9579 | 0.9668 | 0.9475 | 0.9613 | 0.9698 | 0.9591 | 0.6934 | ablation | 4000 | 20260827 | 1000 | True | 0.0100 |
| a1 | 0.9739 | 0.9726 | 0.9701 | 0.9678 | 0.9624 | 0.9739 | 0.9741 | 0.9732 | 0.9737 | 0.9686 | 0.9677 | 0.9581 | 0.9454 | 0.9711 | 0.9692 | 0.9697 | 0.9599 | 0.9620 | 0.9678 | 0.9596 | 0.9667 | 0.9704 | 0.9657 | 0.7076 | ablation | 4000 | 20260827 | 1000 | True | 0.0100 |
| a2 | 0.9739 | 0.9726 | 0.9701 | 0.9678 | 0.9624 | 0.9739 | 0.9741 | 0.9732 | 0.9737 | 0.9686 | 0.9677 | 0.9581 | 0.9454 | 0.9711 | 0.9692 | 0.9697 | 0.9599 | 0.9620 | 0.9678 | 0.9596 | 0.9667 | 0.9704 | 0.9657 | 0.7076 | ablation | 4000 | 20260827 | 1000 | True | 0.0100 |
| a3 | 0.9739 | 0.9728 | 0.9707 | 0.9684 | 0.9631 | 0.9743 | 0.9744 | 0.9732 | 0.9738 | 0.9696 | 0.9678 | 0.9585 | 0.9478 | 0.9708 | 0.9697 | 0.9702 | 0.9610 | 0.9629 | 0.9677 | 0.9606 | 0.9672 | 0.9707 | 0.9663 | 0.6565 | ablation | 4000 | 20260827 | 1000 | True | 0.0100 |
| a7_norecon | 0.9729 | 0.9718 | 0.9698 | 0.9673 | 0.9630 | 0.9734 | 0.9733 | 0.9723 | 0.9727 | 0.9673 | 0.9675 | 0.9581 | 0.9453 | 0.9695 | 0.9681 | 0.9694 | 0.9604 | 0.9615 | 0.9665 | 0.9590 | 0.9661 | 0.9698 | 0.9652 | 0.7061 | ablation | 4000 | 20260827 | 1000 | True | 0.0100 |
