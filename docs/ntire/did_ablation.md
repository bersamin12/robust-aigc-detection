| run | epoch | weights | pool | dim | ntire_val_hard robust | ntire_val_hard clean | ntire_val_hard AUC | ntire_val robust | ntire_val clean | ntire_val AUC | objective |
|---|---|---|---|---|---|---|---|---|---|---|---|
| abl_did_c1ema | 1 | final | mean | 1024 | 0.7930 | 0.9325 | 0.8711 | 0.9304 | 0.9760 | 0.9554 | 0.9080 |
| abl_did_c1ema_teacher | 1 | ema | mean | 1024 | 0.7699 | 0.8933 | 0.8363 | 0.9105 | 0.9531 | 0.9334 | 0.8817 |
| abl_did_c1self | 1 | final | mean | 1024 | 0.7931 | 0.9389 | 0.8755 | 0.9251 | 0.9749 | 0.9519 | 0.9080 |
| abl_did_c1self_a3 | 1 | final | mean | 1024 | 0.7957 | 0.9234 | 0.8670 | 0.9222 | 0.9673 | 0.9466 | 0.9021 |
| abl_pool_mean | 1 | final | mean | 1024 | 0.8040 | 0.9264 | 0.8729 | 0.9280 | 0.9683 | 0.9499 | 0.9067 |

Leaderboard (private test, arXiv 2604.11487): MICV (1st) 0.9723 / 0.9974; Ant International (2nd) 0.9721 / 0.9972; TeleAI-TeleGuard (3rd) 0.9251 / 0.9786 (robust / clean).
