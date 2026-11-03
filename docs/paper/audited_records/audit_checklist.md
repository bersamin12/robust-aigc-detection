# Item-by-item audit checklist

The implementation plan is the supplied representation of both prior audits. The original audit texts/comment IDs were not present in the inspected workspace, so IDs below enumerate every actionable clause of that plan rather than inventing reviewer comment numbers. Status: **addressed**, **already correct**, **unsupported**, or **unverifiable**. See `claim_evidence_map.md` for source paths and exact populations.

| ID | Requested issue | Status | Resolution / evidence gap |
|---|---|---|---|
| S01 | Descriptive exact title; empirical contribution | addressed | New title in both TeX/PDFs; novelty is diagnosis, not resize. |
| S02 | Scope round-trip interpretation to DINOv2 | addressed | CLIP's positive clean round-trip benefit separately reported. |
| S03 | Interpolation, aggregation, token limitations | addressed | Main contrasts, native-control paragraph, limitations; supplement S3/S10. |
| S04 | Trace SocialRF 9.1/7.9 headline | addressed | Exact 0.867656→0.958200 and 0.882892→0.962304 in body. |
| S05 | Distinguish frozen/trained/B-Free SocialRF | addressed | Separate populations and sections; headline has no fabricated paired CI. |
| S06 | Coverage only between 25–100% area on degraded SocialRF | addressed | Abstract, contributions, figure caption, contrasts, public detector, conclusion. |
| S07 | Initial native-quarter gains 5.34/5.10 | addressed | Main body; explicitly point estimates without paired CIs. |
| S08 | Compact four-configuration fusion table | addressed | Main Table 1: 0.50/0.50/0.55/0.45 with fit populations. |
| S09 | Separate z-score estimation from weight selection | addressed | Population SD estimation versus 21-point weight sweep; each fit population stated. |
| S10 | Distinguish freeze/evaluation dates | addressed | Sep 23 freeze vs Oct 4 read; Oct 11 read 2 after validation fit. |
| S11 | Independent exact freeze timestamp for read 2 | unverifiable | Not in available records; disclosed rather than assigning a date. |
| S12 | Explain 200-pixel choice early; preview controls | addressed | Setup cites historical information ceiling/minimum, not optimality. |
| S13 | 200→512 2.56× vs tower 518 2.59× | addressed | Main and supplement; schematic caption. |
| S14 | Native-512 is not entirely interpolation-free | addressed | 512→518 adjustment explicit. |
| S15 | Frozen in-band gains and saved intervals | addressed | 171 images, branchwise +11.1 [5.9,17.0]/+7.9 [3.6,12.9] points. |
| S16 | Mismatch alone inadequate; context interpretive | addressed | Main contrasts and conclusion. |
| S17 | Report degraded B-Free losses as well as gain | addressed | CommunityAI −14.5 points CI excludes zero; Synthbuster −0.4 interval includes zero. |
| S18 | Pooled JPEG95 and degradation CIs | addressed | Main external table adds all saved intervals and pooled JPEG95 AUCs. |
| S19 | Informal leaderboard placement | addressed | Removed. |
| S20 | Exact 630-image population/three 512–196 contrasts | addressed | Main concise paragraph; full supplement S10, single seed. |
| S21 | Clean losses as system differences; hypotheses tentative | addressed | Main exploratory clean cost and S7/S10. |
| P01 | Full-width vector preprocessing schematic | addressed | Setup Fig. 1, native coordinates; two rows and interpolation/count annotations. |
| P02 | Move separately trained size plot to Training | addressed | Plot is there; diagnosis retains frozen evidence. |
| P03 | 4× evaluated only at full coverage | addressed | Main text and factorial plot show isolated marker. |
| P04 | Native and attention-combiner details to supplement | addressed | S10/S11; concise findings and links in main. |
| P05 | Remove R(4) from mixed-population table | addressed | Reported separately with 2,833-source population and saved intervals. |
| P06 | Ten candidates vs two sampled degraded draws | addressed | Main Setup; exact sampler in S3. |
| P07 | Epoch sampling and 224 proxy input | addressed | S3 formula, replacement/proportional sampling, seeds. |
| P08 | Resize kernels/duplicate removal/fov3 below600 | addressed | S3 and main fov3 paragraph; all interpolation stages specified. |
| P09 | B-Free reference preprocessing | addressed | S3 patch-grid/window/replicate-wrap description. Historical demo match labeled reported, not rerun. |
| P10 | Pooled AUC; 50% test distortion share | addressed | Main Setup and Statistics, verification script. |
| P11 | Threshold provenance | addressed | Retrospective population-specific ROC thresholds; pooled fused TPR clarified. |
| P12 | Mark clean-cost analyses exploratory | addressed | Later >1100 declarations preserved with correct chronology. |
| P13 | Multiple-comparison disclaimer | addressed | CIs explicitly unadjusted. |
| P14 | American spelling/policy vocabulary/AUC points | addressed | New files standardized; historical transcriptions retain original wording. |
| P15 | Reconcile repeated numbers; no invented drift cause | addressed | S12 preserves separate passes and gives exact discrepancy; external point estimates agree. |
| P16 | Original “at most 0.0001” probe/rescore statement | already correct | Max observed AUC difference 0.00009472; retained. |
| P17 | Alleged general LaTeX defects | already correct | No original reviewer defect list supplied. Existing ordinary math punctuation/footnotes inspected rather than presumed defective. |
| P18 | Tables, signs, extensions, cross-references | addressed | New source compiled and rendered; signs normalized, PDF extensions explicit, no overfull boxes or unresolved citations. |
| P19 | Quotations against primary text | addressed | Corvi paraphrased after original PDF check; remaining direct quotations removed. |
| P20 | Bibliography and historical crop/augmentation context | addressed | Added AlexNet/Inception/CNNDetection, COCO/Open Images; isolated bibliography. |
| P21 | GlobalForge comparison | addressed | Exact Fig.2 values, both training regimes; balanced accuracy distinct from AUC. |
| P22 | Original dataset sources | addressed | COCO/Open Images/RAISE/Synthbuster cited; AIGIBench attribution retained for its subsets. |
| D01 | Fresh main/supplement TeX/PDF and figures | addressed | New *_reframe_audited files; previous files preserved. |
| D02 | Isolated bibliography and source ZIP | addressed | refs_reframe_audited.bib, standalone rebuild scripts and source ZIP. |
| D03 | Anonymized registrations and provenance ledger | addressed | Full current transcription plus eight historical snapshots, no author/contact metadata. |
| D04 | Independently establish late preregistration | unverifiable | Amendments 7–12 first committed Nov2, reported earlier; missing pre-run logs explicitly disclosed. |
| D05 | Original registrations preserved | addressed | Original hashes checked; originals never edited. |
| D06 | Claim–evidence map and itemized checklist | addressed | These two files and numerical verification artifact. |
| D07 | Branch-A decile result | unsupported | No saved branch-A decile control; only B reported. |
| D08 | Missing paired/TPR intervals | unsupported | Not fabricated; derived differences and TPRs labeled point estimates. |
| D09 | Current SAFE 6–8 pages/deadline | addressed | Official call checked Nov2: 6–8 excluding references, Nov4 11:59pm ET. |
| D10 | Supplemental attachment route | unverifiable | Public submission schema currently exposes only `pdf`; separate supplement/ZIP provided, not claimed uploaded. |
| V01 | pdfLaTeX/BibTeX; unchanged AAAI kit | addressed | Source and style hashes checked, both PDFs built without style edits. |
| V02 | Main content by page8 | addressed | Conclusion ends page8; references start page9. |
| V03 | Fonts/references/overfull boxes | addressed | Automated final checks and visual page review; details in validation.md. |
| V04 | Visual every page and schematic | addressed | Rendered main/supplement, schematic and plots reviewed. |
| V05 | Numbers/settings/populations | addressed | 157 saved-evidence checks plus manual source review; no model or per-image evaluation. |
| V06 | Source ZIP compilation/anonymity/preservation | addressed | Extracted archive rebuilt; metadata and old-artifact hashes checked. |
| V07 | Five-dimension self-review and evidence gaps | addressed | Final audit report plus supplement S13; gaps remain openly listed. |
