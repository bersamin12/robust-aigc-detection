# Primary-source literature verification

Checked November 2, 2026. Only the isolated audited bibliography is used by the new TeX files. Uncited entries from the shared bibliography were omitted. This review checked source identity and claims used in the revised text; it does not assert that every incidental page/DOI field was newly revalidated when already supported by the existing literature record.

| Bibliography key | Primary source | Finding / disposition |
|---|---|---|
| corvi2023detection | [Original paper PDF](https://arxiv.org/pdf/2211.00680) | §2 explains why resizing may erase generation traces. Intro now paraphrases; no extended verbatim quotation. |
| li2025safe | [Authors' article](https://arxiv.org/html/2408.06741) | §4.5 states that resizing diminishes artifacts and center/random crop/full-source test performance is similar for their pipeline. Quotation replaced by paraphrase. |
| grommelt2024fakeorjpeg | [Authors' paper](https://arxiv.org/html/2403.17608) | Object-size caveat for crops is present; quote replaced by paraphrase. Published 2025 proceedings year versus 2024 workshop label is retained from existing publisher record. |
| cui2026globalforge | [Authors' paper](https://arxiv.org/html/2607.14684) | Fig.2 directly inspected. Crop-trained: 67.90 crop, 58.70 resize; resize-trained: 67.10 crop, 67.50 resize. Rows are testing, columns training; metric is balanced accuracy, not AUC. |
| pellegrini2025aigenbench | [Authors' accepted version](https://arxiv.org/html/2504.20865) | Table III verifies DINOv2 fine-tuning 94.24 vs93.44 and CLIP92.04 vs87.66. First pair is resize vs five-crop. DOI is given by authors. |
| li2025aigibench | [Authors' paper](https://arxiv.org/html/2505.12335), [conference version](https://openreview.net/pdf/ec620635cb07c66925e97447622729d2178c09eb.pdf) | Test-time preprocessing comparison and provenance of SocialRF/CommunityAI; current official counts differ from the evaluated copy, so our counts remain specifically attributed to our manifest. NeurIPS2025 Datasets and Benchmarks shown in original PDF. |
| choi2025warpad | [Authors' paper](https://arxiv.org/html/2511.14030), [official proceedings](https://papers.nips.cc/paper_files/paper/2025/hash/1d8f05e4da49a4e1e1b052a3046bceac-Abstract-Conference.html) | §3.2 rescales to a square before patchifying; confirms fixed tile field of view. |
| guillaro2025bfree | [Authors' paper](https://arxiv.org/abs/2412.17671), [reference repository](https://github.com/grip-unina/B-Free) | Detector, five-window reference code, and resized RAISE copy. Local implementation checked statically in public_detector_probe.py; reported historical logit match not rerun. |
| konstantinidou2025texturecrop | [Authors' paper](https://arxiv.org/abs/2407.15500) | Texture-aware native crop selection and high-resolution setting support related-work characterization. |
| karageorgiou2025spai | [Authors' paper](https://arxiv.org/abs/2411.19417) | Arbitrary-resolution spectral representation; CVPR2025 metadata present. |
| mu2026hidanet | [Official ICLR2026 proceedings](https://proceedings.iclr.cc/paper_files/paper/2026/hash/820e42f39773c6cbbd875553db45658f-Abstract-Conference.html) | Official author list matches original five-author bibliography; arXiv2508.17346 has a different author list. Use official proceedings consistently and replace the nonresolving original forum URL. Main now uses only qualitative global/local description, avoiding version-dependent ablation numbers. |
| cozzolino2024raising | [Authors' paper](https://arxiv.org/abs/2312.00195), [released code](https://github.com/grip-unina/ClipBased-SyntheticImageDetection) | Resize-before-crop practice and source identity verified. |
| rajan2025aligned | [Authors' paper](https://arxiv.org/html/2410.11835) | §3 explicitly discusses class-dependent up/downsampling signals introduced by random resized crops. |
| ricker2026biasid | [Authors' paper](https://arxiv.org/abs/2605.31153) | Transformation bias framework identity verified; file-format peculiarity also tied to evaluated-copy diagnostics, not assumed universal. |
| touvron2019fixres | [Authors' paper](https://arxiv.org/abs/1906.06423) | Train/test apparent-scale discrepancy; contextual analogy, not proof of our mechanism. |
| gushchin2026ntire | [Challenge authors' report](https://arxiv.org/abs/2604.11487) | Challenge identity, organizer metrics, and preprocessing context. Removed informal comparison to leaderboard ranks. |
| bammey2024synthbuster | [Original dataset](https://zenodo.org/records/10066460), [author institution's publication record](https://researchportal.ip-paris.fr/fr/publications/synthbuster-towards-detection-of-diffusion-model-generated-images/) | Original record excludes RAISE images; journal volume5, pp1–9, year2024 and DOI match. |
| dangnguyen2015raise | [Original dataset site](https://loki.disi.unitn.it/RAISE/index.php) | Original author list/title/venue/year and distinction between original RAW images and resized benchmark copy. |
| ojha2023univfd | [Authors' paper](https://arxiv.org/abs/2302.10174) | Source identity and original detector; our chance result remains dataset-specific. |
| oquab2024dinov2 | [Authors' paper](https://arxiv.org/abs/2304.07193) | Backbone identity; existing TMLR reference retained. |
| darcet2024registers | [Authors' paper](https://arxiv.org/abs/2309.16588) | Registers identity and authors. |
| izmailov2018swa | [Authors' paper](https://arxiv.org/abs/1803.05407) | Weight-averaging method identity. |
| balle2018hyperprior | [Authors' paper](https://arxiv.org/abs/1802.01436) | Neural-codec reference identity. |
| radford2021clip | [Authors' paper](https://arxiv.org/abs/2103.00020) | CLIP source identity; no claim our normalization is the original CLIP normalization. |
| han2026glass | [Authors' paper](https://arxiv.org/abs/2601.00141) | Global view plus multiple local crops; future-work context. |
| krizhevsky2012imagenet | [Original NeurIPS paper](https://proceedings.neurips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html) | Historical random-crop/multi-crop inference context, newly cited. |
| szegedy2015going | [Authors' paper](https://arxiv.org/abs/1409.4842) | Historical scale/aspect crop augmentation context, newly cited. |
| wang2020cnn | [Official CVPR2020 proceedings](https://openaccess.thecvf.com/content_CVPR_2020/html/Wang_CNN-Generated_Images_Are_Surprisingly_Easy_to_Spot..._for_Now_CVPR_2020_paper.html) | Historical forensic blur/JPEG augmentation context, newly cited. |
| lin2014coco | [Original dataset paper](https://arxiv.org/abs/1405.0312) | Original COCO attribution added. |
| kuznetsova2020openimages | [Original dataset paper](https://arxiv.org/abs/1811.00982) | Original Open Images attribution added. |

No directly quoted text remains in the audited main paper. The historical registration transcriptions preserve their own wording and quoted decision rules as user-owned research records; they are not new quotations from external papers. The GlobalForge figure was inspected from its original image, not inferred from prose. No causal conclusions were borrowed from another detector's ablation.
