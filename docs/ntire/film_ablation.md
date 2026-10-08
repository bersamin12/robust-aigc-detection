| run | epoch | weights | pool | dim | ntire_val_hard robust | ntire_val_hard clean | ntire_val_hard AUC | ntire_val robust | ntire_val clean | ntire_val AUC | objective |
|---|---|---|---|---|---|---|---|---|---|---|---|
| abl_film_film | 1 | final | mean | 1024 | 0.8032 | 0.9252 | 0.8717 | 0.9290 | 0.9673 | 0.9496 | 0.9062 |
| abl_film_film_shuf | 1 | final | mean | 1024 | 0.6871 | 0.7633 | 0.7279 | 0.7843 | 0.7973 | 0.7896 | 0.7580 |
| abl_pool_mean | 1 | final | mean | 1024 | 0.8040 | 0.9264 | 0.8729 | 0.9280 | 0.9683 | 0.9499 | 0.9067 |

Leaderboard (private test, arXiv 2604.11487): MICV (1st) 0.9723 / 0.9974; Ant International (2nd) 0.9721 / 0.9972; TeleAI-TeleGuard (3rd) 0.9251 / 0.9786 (robust / clean).
