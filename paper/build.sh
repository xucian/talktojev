#!/bin/bash
# Numbers from the runs, then the checks, then the PDF, then the arXiv upload set. Any failure stops the build.
#   brew install tectonic   (one self-contained binary; brew uninstall tectonic removes it)
set -euo pipefail
cd "$(dirname "$0")/.."
python paper/make_numbers.py >/dev/null
python paper/check.py
cd paper
rm -f paper.aux paper.bbl  # left by --keep-intermediates; a stale pair makes the first pass warn about lines the final pass fixes
tectonic -X compile --keep-intermediates paper.tex 2>&1 | tee build.log | grep -iE "warning|error|undefined|overfull" || true
if grep -qiE "undefined (reference|citation)|error:" build.log; then echo "[paper] build has undefined references or errors"; exit 1; fi
test -s paper.bbl || { echo "[paper] no paper.bbl (arXiv does not run BibTeX)"; exit 1; }
# arXiv needs the .bbl (it does not run BibTeX) and \pdfoutput=1 in the first lines to pick pdflatex
# for the PNG figure. XeTeX (tectonic) has no \pdfoutput, so the line goes into the uploaded copy only.
rm -rf arxiv && mkdir -p arxiv/figures
{ printf '%s\n' '\pdfoutput=1'; cat paper.tex; } > arxiv/paper.tex
cp numbers.tex author.txt refs.bib paper.bbl arxiv/
cp figures/conversation.png arxiv/figures/
tar czf arxiv.tar.gz -C arxiv paper.tex numbers.tex author.txt refs.bib paper.bbl figures
echo "[paper] paper/paper.pdf and paper/arxiv.tar.gz"
