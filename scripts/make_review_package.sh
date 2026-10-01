#!/usr/bin/env bash
# Build the anonymised supplementary package for double-blind review.
#   usage: scripts/make_review_package.sh [path/to/rubricon]   (default ../rubricon)
# Output: submission/anonymised_supplementary.zip (gitignored).
# The script fails if any identifying string survives, and it checks that the
# renamed code still installs and passes its tests.
set -euo pipefail
cd "$(dirname "$0")/.."
LIB=${1:-../rubricon}
OUT=submission/anon
rm -rf "$OUT" submission/anonymised_supplementary.zip
mkdir -p "$OUT/code/anonlib"
git archive HEAD | tar -x -C "$OUT/code"
(cd "$LIB" && git archive HEAD) | tar -x -C "$OUT/code/anonlib"
cd "$OUT/code"

# 1. Files that identify the author or exist only for the public release.
rm -rf AUTHOR_TODO.md POST.md POST-social.md CITATION.cff .zenodo.json .github paper/preprint.tex \
       scripts/make_review_package.sh \
       anonlib/CITATION.cff anonlib/.zenodo.json anonlib/.github anonlib/docs/rubricon-walkthrough.pptx

# 2. Preprint-only LaTeX branches (author block, code availability) and the library name.
python3 - <<'PY'
import re
from pathlib import Path
for p in [Path("paper/main.tex"), Path("paper/short.tex")]:
    s = p.read_text()
    s = re.sub(r"\\ifdefined\\PREPRINT\n\\author\{.*?\}\n\\else\n(\\author\{Anonymous submission\})\n\\fi", r"\1", s, flags=re.S)
    s = re.sub(r"\\ifdefined\\PREPRINT\n\\section\*\{Code and data availability\}.*?\\fi\n", "", s, flags=re.S)
    s = re.sub(r"\\ifdefined\\PREPRINT\\usepackage\[preprint\]\{acl\}\\else(\\usepackage\[review\]\{acl\})\\fi", r"\1", s)
    s = re.sub(r"\\ifdefined\\PREPRINT\\newcommand\{\\libname\}\{Rubricon\}(\\newcommand\{\\libintro\}\{[^}]*\})?\\else(.*?)\\fi", r"\2", s)
    s = s.replace("% preprint.tex defines \\PREPRINT to build the non-anonymous arXiv version from this file.\n", "")
    p.write_text(s)
PY

# 3. Rename the library everywhere (package, imports, docs), then scrub links and names.
mv anonlib/src/rubricon anonlib/src/anonlib
mv src/rubricon_field src/anonlib_field
LC_ALL=C grep -rIl -i -e rubricon . | while read -r f; do
  sed -i '' -e 's/rubricon/anonlib/g' -e 's/Rubricon/AnonLib/g' -e 's/RUBRICON/ANONLIB/g' "$f"
done
LC_ALL=C grep -rIl -iE "yogvid|wankhede|github\.com/|zenodo|orcid" . | while read -r f; do
  sed -i '' -E \
    -e '/\[!\[(ci|release|license|OpenSSF|DOI)/d' \
    -e 's#https?://github\.com/yogvidwankhede/[A-Za-z0-9_.-]+#[anonymised repository]#g' \
    -e 's#https?://yogvidwankhede\.github\.io/[A-Za-z0-9_./-]+#[anonymised]#g' \
    -e 's#https?://doi\.org/10\.5281/zenodo\.[0-9]+#[anonymised archive]#g' \
    -e 's#10\.5281/zenodo\.[0-9]+#[anonymised archive]#g' \
    -e 's#\(https://github\.com/yogvidwankhede\)##g' \
    -e 's/Yogvid Wankhede/Anonymous/g; s/Wankhede/Anonymous/g; s/Yogvid/Anonymous/g; s/yogvidwankhede/anonymous/g' \
    "$f"
done
sed -i '' -E 's/as of commit `[0-9a-f]+`/as of commit `[anonymised]`/' CLAIMS.md

# 4. Fail if anything identifying survived (text files, and strings inside binaries).
if LC_ALL=C grep -rIn -iE "yogvid|wankhede|wustl|washington university|washu|gmail|0009-0004|zenodo\.[0-9]{5,}|rubricon|github\.com/yogvid" . ; then
  echo "IDENTIFYING STRING FOUND -- aborting" >&2; exit 1
fi
if find . -type f \( -name '*.pdf' -o -name '*.png' -o -name '*.parquet' -o -name '*.pptx' \) -print0 | xargs -0 strings 2>/dev/null | grep -iqE "yogvid|wankhede"; then
  echo "IDENTIFYING STRING IN A BINARY -- aborting" >&2; exit 1
fi
cd ../..
cp ../paper/main.pdf anon/paper_long_review.pdf
cp ../paper/short.pdf anon/paper_short_review.pdf
for f in anon/*.pdf; do
  if pdftotext "$f" - | grep -iqE "yogvid|wankhede|rubricon|zenodo\.[0-9]|github\.com/yogvid"; then echo "IDENTIFYING TEXT IN $f" >&2; exit 1; fi
done
cat > anon/README_ANON.md <<'MD'
# Anonymised supplementary material

- `paper_long_review.pdf`, `paper_short_review.pdf`: the submission PDFs.
- `code/`: the reproducibility package (analysis, results, paper sources). See `code/README.md`.
  - `make all` regenerates every number, table and figure from the public corpora without calling any model API.
  - `code/anonlib/` is the statistics library (its real name is withheld for review). Install it with `pip install -e code/anonlib`.
- `code/CLAIMS.md` maps every number in the paper to its results file, JSON path and producing script.
- `code/PREREG.md`, `code/PREREG_ADDENDUM.md` and `code/DEVIATIONS.md` hold the pre-registered plan and every departure from it.

No corpus text is included. `code/scripts/fetch_corpora.sh` downloads each corpus from its original host and verifies its checksum.
MD
(cd anon && zip -qr ../anonymised_supplementary.zip .)
echo "built submission/anonymised_supplementary.zip ($(du -h anonymised_supplementary.zip | cut -f1))"
