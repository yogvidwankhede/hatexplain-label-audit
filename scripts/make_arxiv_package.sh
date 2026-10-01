#!/usr/bin/env bash
# Build the non-anonymous arXiv source package: submission/arxiv_source.tar.gz (gitignored).
set -euo pipefail
cd "$(dirname "$0")/.."
(cd paper && tectonic -X compile --keep-intermediates preprint.tex >/dev/null)
A=submission/arxiv; rm -rf "$A"; mkdir -p "$A/figures" "$A/tables"
cp paper/{acl.sty,acl_natbib.bst,numbers.tex,judges_section.tex,judges_appendix.tex,appendix.tex,references.bib} "$A/"
cp paper/figures/*.pdf "$A/figures/"; cp paper/tables/*.tex "$A/tables/"
{ echo '\def\PREPRINT{}'; cat paper/main.tex; } > "$A/main.tex"
cp paper/preprint.bbl "$A/main.bbl"
(cd "$A" && tectonic -X compile main.tex >/dev/null)   # must compile standalone
tar -czf submission/arxiv_source.tar.gz -C "$A" --exclude main.pdf --exclude '*.aux' --exclude '*.log' .
cp paper/preprint.pdf submission/preprint_nonanonymous.pdf
echo "built submission/arxiv_source.tar.gz ($(du -h submission/arxiv_source.tar.gz | cut -f1))"
