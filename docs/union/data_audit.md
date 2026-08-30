# Pre-normalisation data audit

| source | label | n | fmt_top | mode_top | width_median | height_median | jpeg_q_median |
| --- | --- | --- | --- | --- | --- | --- | --- |
| coco_train2017 | 0 | 39924 | JPEG | RGB | 640.0 | 480.0 | 96.05901766647253 |
| ntire | 0 | 54000 | JPEG | RGB | 1072.0 | 1024.0 | 95.0 |
| ntire | 1 | 95999 | JPEG | RGB | 1024.0 | 960.0 | 95.0 |
| open_images | 0 | 25000 | JPEG | RGB | 427.0 | 640.0 | 90.0 |
| sid_set | 0 | 29317 | PNG | RGB | 1024.0 | 768.0 | 83.07144166971273 |
| sid_set | 1 | 29438 | PNG | RGB | 1024.0 | 1024.0 | 90.85016754293841 |
| wildfake | 0 | 40000 | JPEG | RGB | 200.0 | 200.0 | 75.89285714285714 |
| wildfake | 1 | 61680 | PNG | RGB | 256.0 | 256.0 | 88.17637680131973 |

## Flags

- Resolution confound: median width 640 vs 1024
- Resolution confound: source 'coco_train2017' (authentic) median width 640 vs generated class 1024
- Resolution confound: source 'ntire' (generated) median width 1024 vs authentic class 640
- Resolution confound: source 'open_images' (authentic) median width 427 vs generated class 1024
- Resolution confound: source 'sid_set' (generated) median width 1024 vs authentic class 640
- Resolution confound: source 'wildfake' (authentic) median width 200 vs generated class 1024
- JPEG-quality confound: source 'wildfake' (authentic) median q 76 vs generated class 91
- Resolution confound: source 'wildfake' (generated) median width 256 vs authentic class 640
- Resolution heterogeneity within authentic: source 'coco_train2017' median width 640 vs 'ntire' 1072
- Format heterogeneity within authentic: source 'coco_train2017' is JPEG but 'sid_set' is PNG
- Resolution heterogeneity within authentic: source 'coco_train2017' median width 640 vs 'sid_set' 1024
- JPEG-quality heterogeneity within authentic: source 'coco_train2017' median q 96 vs 'sid_set' 83
- Resolution heterogeneity within authentic: source 'coco_train2017' median width 640 vs 'wildfake' 200
- JPEG-quality heterogeneity within authentic: source 'coco_train2017' median q 96 vs 'wildfake' 76
- Resolution heterogeneity within authentic: source 'ntire' median width 1072 vs 'open_images' 427
- Format heterogeneity within authentic: source 'ntire' is JPEG but 'sid_set' is PNG
- JPEG-quality heterogeneity within authentic: source 'ntire' median q 95 vs 'sid_set' 83
- Resolution heterogeneity within authentic: source 'ntire' median width 1072 vs 'wildfake' 200
- JPEG-quality heterogeneity within authentic: source 'ntire' median q 95 vs 'wildfake' 76
- Format heterogeneity within authentic: source 'open_images' is JPEG but 'sid_set' is PNG
- Resolution heterogeneity within authentic: source 'open_images' median width 427 vs 'sid_set' 1024
- Resolution heterogeneity within authentic: source 'open_images' median width 427 vs 'wildfake' 200
- JPEG-quality heterogeneity within authentic: source 'open_images' median q 90 vs 'wildfake' 76
- Format heterogeneity within authentic: source 'sid_set' is PNG but 'wildfake' is JPEG
- Resolution heterogeneity within authentic: source 'sid_set' median width 1024 vs 'wildfake' 200
- Format heterogeneity within generated: source 'ntire' is JPEG but 'sid_set' is PNG
- Format heterogeneity within generated: source 'ntire' is JPEG but 'wildfake' is PNG
- Resolution heterogeneity within generated: source 'ntire' median width 1024 vs 'wildfake' 256
- Resolution heterogeneity within generated: source 'sid_set' median width 1024 vs 'wildfake' 256