| run | epoch | weights | pool | dim | ntire_val robust | ntire_val clean | ntire_val AUC | ntire_val_hard robust | ntire_val_hard clean | ntire_val_hard AUC | val robust | val clean | val AUC | objective |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A2 | 2 | final | mean | 1024 | 0.9617 | 0.9968 | 0.9814 | 0.8561 | 0.9913 | 0.9408 | n/a | 0.9999 | 0.9999 | 0.9515 |
| A2swa | 2 | swa | mean | 1024 | 0.9609 | 0.9967 | 0.9810 | 0.8541 | 0.9912 | 0.9396 |  |  |  | 0.9507 |
| B2 | 3 | final | mean | 2048 | 0.9660 | 0.9909 | 0.9793 | 0.8853 | 0.9800 | 0.9411 | n/a | 0.9998 | 0.9998 | 0.9556 |
| B2swa | 3 | swa | mean | 2048 | 0.9669 | 0.9911 | 0.9798 | 0.8873 | 0.9801 | 0.9420 |  |  |  | 0.9564 |

Leaderboard (private test, arXiv 2604.11487): MICV (1st) 0.9723 / 0.9974; Ant International (2nd) 0.9721 / 0.9972; TeleAI-TeleGuard (3rd) 0.9251 / 0.9786 (robust / clean).
