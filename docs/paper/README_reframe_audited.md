# Audited SAFE package

Use `main_reframe_audited.pdf` as the main submission and keep `supplement_reframe_audited.pdf` separate. These are new versions based on `main_reframe_revised.tex`; prior drafts, figures, bibliography, registration records, and README are preserved.

The [official SAFE call](https://www.safeworkshop.org/aaai-2027/) checked November 2, 2026 specifies **6–8 pages excluding references**, with a deadline of **November 4, 2026, 11:59 pm ET** (November 5, 12:59 pm Singapore). Main content ends on page 8. The public [OpenReview submission](https://openreview.net/group?id=AAAI.org/2027/Workshop/SAFE) currently exposes only a main-PDF file field; the attachment route for the separate supplement and source ZIP needs confirmation from the venue. No files have been submitted or attached by this task.

Build locally with the unchanged AAAI style:

```bash
bash build_reframe_audited.sh
```

Requires pdfLaTeX, BibTeX, and the AAAI style's font packages. The source ZIP includes the two TeX files, an isolated bibliography, matching figure PDFs, the unchanged `aaai2027.sty/.bst`, and anonymized evidence records. `build_reframe_audited.sh` writes build intermediates under `build_audited/`, preserving other versions.

Audit entry points:

- [Audit report](audited_records/audit_report.md), [itemized checklist](audited_records/audit_checklist.md), and [claim–evidence map](audited_records/claim_evidence_map.md).
- [Validation](audited_records/validation.md), [provenance ledger](audited_records/provenance_ledger.md), and [literature checks](audited_records/literature_verification.md).
- `python audited_records/verify_saved_evidence.py`: verifies headline/table numbers and populations using the packaged reports, without loading images/models.
- `python audited_records/generate_assets.py`: redraws vector assets from saved JSON (requires Matplotlib).

Evidence gaps remain: no branch-A decile control or missing TPR/paired intervals can be supplied; late declaration timing is not independently established; numerical drift between some passes is unexplained; clean losses and resizing mechanisms remain confounded. Packaging registrations now does not prove historical preregistration.
