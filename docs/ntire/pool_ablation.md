| run | epoch | weights | pool | dim | ntire_val_hard robust | ntire_val_hard clean | ntire_val_hard AUC | ntire_val robust | ntire_val clean | ntire_val AUC | objective |
|---|---|---|---|---|---|---|---|---|---|---|---|
| abl_pool_attn | 1 | final | attn | 1024 | 0.7916 | 0.9170 | 0.8618 | 0.9240 | 0.9652 | 0.9461 | 0.8995 |
| abl_pool_cls | 1 | final | cls | 1024 | 0.8040 | 0.9243 | 0.8710 | 0.9255 | 0.9685 | 0.9486 | 0.9056 |
| abl_pool_cls_mean | 1 | final | cls_mean | 2048 | 0.8040 | 0.9233 | 0.8705 | 0.9283 | 0.9671 | 0.9491 | 0.9057 |
| abl_pool_cls_reg_mean | 1 | final | cls_reg_mean | 6144 | 0.8042 | 0.9222 | 0.8700 | 0.9278 | 0.9675 | 0.9489 | 0.9054 |
| abl_pool_mean | 1 | final | mean | 1024 | 0.8040 | 0.9264 | 0.8729 | 0.9280 | 0.9683 | 0.9499 | 0.9067 |

Leaderboard (private test, arXiv 2604.11487): MICV (1st) 0.9723 / 0.9974; Ant International (2nd) 0.9721 / 0.9972; TeleAI-TeleGuard (3rd) 0.9251 / 0.9786 (robust / clean).
