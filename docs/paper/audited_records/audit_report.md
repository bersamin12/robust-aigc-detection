# Audited SAFE submission package

The new version implements the supplied plan using saved JSON/Markdown evidence and static code inspection. It does not train, infer, load model checkpoints, recompute per-image AUCs, or reevaluate the test split. Previous drafts and original registrations remain unchanged.

The main paper is an empirical analysis with the requested descriptive title. Its exact SocialRF frozen-checkpoint values are traceable in the body; coverage claims are restricted to 25–100% image area on degraded SocialRF and accompanied by the earlier 5.34/5.10-point gains from one crop to central-quarter tiling. Four fusion configurations and separate normalization/weight fits are explicit. Frozen in-band controls, native-crop contrasts, pooled JPEG95 results, degradation intervals, and B-Free failures are included. Clean-cost localization is exploratory and causal explanations remain hypotheses.

The main has eight pages of content followed by references. The supplement contains moved crop/combiner details, failed controls, threshold provenance, numerical reconciliation, and an anonymized registration appendix. The full current registration and eight historical snapshots are packaged alongside hashes and the provenance ledger. Late amendments' reported pre-run dates cannot be independently established from their first available November2 commit; this package makes that limitation explicit.

Read the [itemized checklist](audit_checklist.md) for each plan clause and the [claim–evidence map](claim_evidence_map.md) for exact sources. [Numerical verification](numeric_verification.json) records 157 checks; [validation](validation.md) records compilation, fonts, layout, archive rebuild, anonymity, and preservation.

## Five-dimension self-review

| Dimension and question | Answer | Assessment |
|---|---|---|
| Contribution: what new knowledge is supported beyond established resizing? | Equal input pixels can hide substantially different crop support. Frozen interventions, factorial contrasts, and external failures show where downscaling helps and where whole-image presentation costs clean performance. | Pass as empirical analysis; no new resize algorithm or uniquely identified mechanism claimed. |
| Clarity: can the configurations, populations, and measurements be reconstructed? | Canonical versus tower size, candidate versus sampled views, kernels, epoch draws, fusion fits, and ROC threshold provenance are explicit; exact saved sources are mapped. | Pass for reporting; original scores/checkpoints are not bundled, so this is not an independent experiment reproduction. |
| Experimental strength: are gains meaningful and failures visible? | Frozen SocialRF gains are 9.1/7.9 points, with separate within-band intervals. B-Free CommunityAI losses, Synthbuster null/loss, clean costs, and failed masking/routing are retained. | Pass for measured effects; selection-split optimism and synthetic degradations limit generalization. |
| Evaluation completeness: what evidence is missing? | There is no saved A-branch decile control, no CI for several derived differences/TPRs, no late pre-run logs, and only a single seed for 512/196. These gaps are marked instead of filled. | Needs future evidence for broader claims; no additional experiment was authorized or run. |
| Design soundness: is field of view causally isolated? | No. Interpolation, aggregation, token counts, training augmentation scale, and degradation scale co-vary. Round-trip residuals are diagnostic contrasts; context is an interpretation. | Pass with narrowed claims; identifying causal components needs another design. |

## Remaining evidence gaps

- Branch-A native-size-decile results and missing paired/TPR confidence intervals do not exist in the supplied records. No values or intervals were fabricated.
- Local Git history verifies declaration content at commits, but does not independently certify run ordering. Amendments7–12 first appear November2; earlier text dates and historical staged-diff claims lack the required pre-run logs here.
- The exact independent read-2 configuration-freeze timestamp is missing. Evaluation dates and reported freeze dates are distinguished.
- Small probe/as-trained cross-pass AUC drift remains unexplained. The original ≤0.0001 bound was correct. External standalone fusion and paired-report differences agree to numerical precision.
- Global context, spectral-trace loss, and generator-pipeline explanations remain hypotheses. Clean losses compare systems with other differences; the native-control 196 input also changes tokens and augmentation scale.
- The supplied plan summarizes two audits, but the original audit texts/IDs were unavailable. The checklist covers every actionable plan clause without inventing missing reviewer comments.
- The present OpenReview form has a main-PDF field only. The supplement and source archive are local deliverables, not attached submissions; their submission route remains to be confirmed.
