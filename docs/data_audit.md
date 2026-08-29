# Pre-normalisation data audit

| source | label | n | fmt_top | mode_top | width_median | height_median | jpeg_q_median |
| --- | --- | --- | --- | --- | --- | --- | --- |
| coco_val2017 | 0 | 5000 | JPEG | RGB | 640.0 | 480.0 | 96.05901766647253 |
| sid_set | 0 | 10049 | PNG | RGB | 1024.0 | 768.0 | 83.06492588960107 |
| sid_set | 1 | 10079 | PNG | RGB | 1024.0 | 1024.0 | 90.8478364812777 |
| wildfake | 0 | 55000 | JPEG | RGB | 200.0 | 200.0 | 75.89285714285714 |
| wildfake | 1 | 62988 | PNG | RGB | 256.0 | 256.0 | 88.2031676126679 |

## Flags

- Format confound: authentic ['JPEG', 'PNG'] vs generated ['PNG']
- Format confound: source 'coco_val2017' (authentic) is JPEG but no generated source is
- Resolution confound: source 'sid_set' (authentic) median width 1024 vs generated class 640
- Resolution confound: source 'sid_set' (generated) median width 1024 vs authentic class 640
- Format confound: source 'wildfake' (authentic) is JPEG but no generated source is
- Resolution confound: source 'wildfake' (authentic) median width 200 vs generated class 640
- JPEG-quality confound: source 'wildfake' (authentic) median q 76 vs generated class 90
- Resolution confound: source 'wildfake' (generated) median width 256 vs authentic class 640
- Format heterogeneity within authentic: source 'coco_val2017' is JPEG but 'sid_set' is PNG
- Resolution heterogeneity within authentic: source 'coco_val2017' median width 640 vs 'sid_set' 1024
- JPEG-quality heterogeneity within authentic: source 'coco_val2017' median q 96 vs 'sid_set' 83
- Resolution heterogeneity within authentic: source 'coco_val2017' median width 640 vs 'wildfake' 200
- JPEG-quality heterogeneity within authentic: source 'coco_val2017' median q 96 vs 'wildfake' 76
- Format heterogeneity within authentic: source 'sid_set' is PNG but 'wildfake' is JPEG
- Resolution heterogeneity within authentic: source 'sid_set' median width 1024 vs 'wildfake' 200
- Resolution heterogeneity within generated: source 'sid_set' median width 1024 vs 'wildfake' 256