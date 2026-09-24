#!/bin/bash
# Build the paper: numbers from the runs, the checks, the PDF (paper/paper.pdf, served at /paper.pdf after a deploy).
#   ./paper.build.sh            build
#   ./paper.build.sh --open     build, then open the PDF
# Needs tectonic:  brew install tectonic
set -euo pipefail
cd "$(dirname "$0")"
command -v tectonic >/dev/null || { echo "[paper] tectonic missing: brew install tectonic"; exit 1; }
./paper/build.sh
[ "${1:-}" = "--open" ] && open paper/paper.pdf
echo "[paper] next: ./deploy.prod.remotely.sh ships it; ./publish.sh --paper refreshes the source repo"
