| run | epoch | weights | pool | dim | ntire_val_hard robust | ntire_val_hard clean | ntire_val_hard AUC | ntire_val robust | ntire_val clean | ntire_val AUC | objective |
|---|---|---|---|---|---|---|---|---|---|---|---|
| abl_crop_s1 | 1 | final | mean | 1024 | 0.8037 | 0.9264 | 0.8728 | 0.9279 | 0.9683 | 0.9498 | 0.9066 |
| abl_crop_s2 | 1 | final | mean | 1024 | 0.7952 | 0.9266 | 0.8693 | 0.9316 | 0.9691 | 0.9519 | 0.9056 |
| abl_crop_s3 | 1 | final | mean | 1024 | 0.7993 | 0.9234 | 0.8681 | 0.9298 | 0.9693 | 0.9514 | 0.9055 |
| abl_squish_s1 | 1 | final | mean | 1024 | 0.8866 | 0.9346 | 0.9115 | 0.9648 | 0.9792 | 0.9726 | 0.9413 |
| abl_squish_s2 | 1 | final | mean | 1024 | 0.8881 | 0.9384 | 0.9139 | 0.9663 | 0.9811 | 0.9743 | 0.9435 |
| abl_squish_s3 | 1 | final | mean | 1024 | 0.8776 | 0.9342 | 0.9073 | 0.9666 | 0.9810 | 0.9744 | 0.9399 |

Leaderboard (private test, arXiv 2604.11487): MICV (1st) 0.9723 / 0.9974; Ant International (2nd) 0.9721 / 0.9972; TeleAI-TeleGuard (3rd) 0.9251 / 0.9786 (robust / clean).
