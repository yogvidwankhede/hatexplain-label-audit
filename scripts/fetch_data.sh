#!/usr/bin/env bash
# Fetch the HateXplain corpus. Not vendored: it is 12MB and belongs to its authors.
# Mathew et al. (2021), HateXplain: A Benchmark Dataset for Explainable Hate Speech
# Detection. AAAI 35(17), 14867-14875. Source: github.com/hate-alert/HateXplain (MIT).
set -euo pipefail
mkdir -p data
curl -fsSL -o data/hatexplain.json \
  https://raw.githubusercontent.com/hate-alert/HateXplain/master/Data/dataset.json
python3 -c "import json;d=json.load(open('data/hatexplain.json'));assert len(d)==20148,len(d);print(f'ok: {len(d)} posts')"
