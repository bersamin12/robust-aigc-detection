#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p build_audited
cp aaai2027.bst refs_reframe_audited.bib build_audited/
for doc in main_reframe_audited supplement_reframe_audited; do
  pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build_audited "$doc.tex" > "build_audited/$doc.pass1.stdout"
  (cd build_audited && bibtex "$doc" > "$doc.bibtex.stdout")
  pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build_audited "$doc.tex" > "build_audited/$doc.pass2.stdout"
  pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build_audited "$doc.tex" > "build_audited/$doc.pass3.stdout"
  if rg -q 'Overfull|undefined references|undefined citations|Citation .* undefined|Reference .* undefined' "build_audited/$doc.log"; then
    echo "Unresolved layout/reference problem: build_audited/$doc.log" >&2
    exit 1
  fi
  cp "build_audited/$doc.pdf" "$doc.pdf"
done
