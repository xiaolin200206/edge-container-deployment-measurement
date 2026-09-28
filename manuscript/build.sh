#!/bin/sh
# Regenerate every number, table and figure from the raw run logs, then build
# the paper and the supplemental material. Run from anywhere:
#   sh manuscript/build.sh
# Needs: python3 with pandas, numpy, matplotlib; a TeX Live with IEEEtran.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
TOP="$(dirname "$HERE")"
cd "$TOP"
python3 analysis/analyse_v2.py
python3 analysis/make_tex.py
python3 analysis/make_supp.py
python3 analysis/figures_v2.py
cp out/numbers_v2.tex out/table_results.tex out/supp_tables.tex "$HERE/"
mkdir -p "$HERE/figures" && cp figures/*.pdf "$HERE/figures/"
cd "$HERE"
for doc in main supplement; do
  pdflatex -interaction=nonstopmode $doc.tex >/dev/null
  [ $doc = main ] && bibtex main >/dev/null
  pdflatex -interaction=nonstopmode $doc.tex >/dev/null
  pdflatex -interaction=nonstopmode $doc.tex >/dev/null
done
grep -E "Overfull|undefined" main.log supplement.log || true
echo "main.pdf: $(pdfinfo main.pdf | awk '/Pages/{print $2}') pages"
