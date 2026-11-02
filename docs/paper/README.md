# Paper draft

- `main.tex`: the submission draft for SAFE @ AAAI 2027 (deadline 2026-11-02
  AoE), AAAI-27 author kit (`aaai2027.sty`, `aaai2027.bst`, template version
  2027.1, from https://aaai.org/authorkit27/). Anonymous mode.
- `main_cvpr.tex`: the same text in the CVPR template, kept for a CVPR 2027
  workshop route. It needs the CVPR author kit, which is not in this folder.
- `refs.bib`: bibliography. Fields not checked against the publisher are
  listed at its top.
- `related.md`: the literature and novelty check behind the related-work
  section.
- `numbers.md`: the source file of every number in the paper.
- `main_reframe.tex`, `supplement_reframe.tex`: the reframed draft (2026-11-02),
  one story around field of view, with Tests 13-21 in. Not yet agreed by the
  group, so `main.tex` is unchanged. Uses `fig_tiles.pdf`.
- `preview_reframe.pdf`, `preview_reframe_supplement.pdf`: tectonic builds of
  the reframe, for reading only. `preview.pdf` is still the old `main.tex`.

## Building

`aaai2027.sty` requires pdfLaTeX:

    pdflatex main && bibtex main && pdflatex main && pdflatex main

Overleaf works. For a quick length check on a machine with only tectonic
(XeTeX), compile a COPY of this folder with the style's `\RequirePDFTeX` line
commented out. The page geometry is the same, but fonts can differ slightly
from pdfLaTeX, so the submission PDF must come from pdfLaTeX.

`\pending{...}` marks text that waits on a running experiment. It must be gone
before submission, along with `\usepackage{xcolor}`, which exists only for it.
