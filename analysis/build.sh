#!/bin/sh
# Regenerate every number, table and figure of the three-condition study from
# the raw run logs. Run from anywhere:
#   sh analysis/build.sh
# Needs: python3 with pandas, numpy, matplotlib.
# Output: out/numbers_v2.json (all quantities), out/numbers_v2.tex (one LaTeX
# macro per reported value), out/table_*.tex and out/supp_tables.tex (tables),
# figures/fig1..fig7 (.pdf, .png); out/extras_v2.json, out/extras_v2.tex,
# out/table_cpu_energy.tex and out/table_extras_runs.tex (derived quantities). The .tex files use only standard LaTeX
# (booktabs, amsmath), so they can be \input into any journal template.
set -e
TOP="$(cd "$(dirname "$0")/.." && pwd)"
cd "$TOP"
python3 analysis/analyse_v2.py
python3 analysis/make_tex.py
python3 analysis/make_supp.py
python3 analysis/figures_v2.py
python3 analysis/extras_v2.py
python3 analysis/make_tex_extras.py
echo "done: out/ and figures/ regenerated"
