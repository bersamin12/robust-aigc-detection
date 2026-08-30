# Pre-normalisation data audit

| source | label | n | fmt_top | mode_top | width_median | height_median | jpeg_q_median |
| --- | --- | --- | --- | --- | --- | --- | --- |
| coco_train2017 | 0 | 46715 | JPEG | RGB | 640.0 | 480.0 | 96.05901766647253 |
| sid_set | 0 | 29317 | PNG | RGB | 1024.0 | 768.0 | 83.07144166971273 |
| sid_set | 1 | 29438 | PNG | RGB | 1024.0 | 1024.0 | 90.85016754293841 |
| wildfake | 0 | 15000 | JPEG | RGB | 800.0 | 768.0 | 75.0 |
| wildfake | 1 | 61680 | PNG | RGB | 256.0 | 256.0 | 88.17637680131973 |

## Flags

- Format confound: authentic ['JPEG', 'PNG'] vs generated ['PNG']
- Format confound: source 'coco_train2017' (authentic) is JPEG but no generated source is
- Resolution confound: source 'sid_set' (authentic) median width 1024 vs generated class 640
- Format confound: source 'wildfake' (authentic) is JPEG but no generated source is
- JPEG-quality confound: source 'wildfake' (authentic) median q 75 vs generated class 90
- Resolution confound: source 'wildfake' (generated) median width 256 vs authentic class 800
- Format heterogeneity within authentic: source 'coco_train2017' is JPEG but 'sid_set' is PNG
- Resolution heterogeneity within authentic: source 'coco_train2017' median width 640 vs 'sid_set' 1024
- JPEG-quality heterogeneity within authentic: source 'coco_train2017' median q 96 vs 'sid_set' 83
- JPEG-quality heterogeneity within authentic: source 'coco_train2017' median q 96 vs 'wildfake' 75
- Format heterogeneity within authentic: source 'sid_set' is PNG but 'wildfake' is JPEG
- Resolution heterogeneity within generated: source 'sid_set' median width 1024 vs 'wildfake' 256