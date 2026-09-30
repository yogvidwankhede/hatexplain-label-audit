#!/usr/bin/env bash
# Fetch the additional per-annotator corpora used in the cross-corpus audit.
# Data is NOT vendored (see .gitignore). Licences were read from the primary sources on 2026-09-29:
#   Measuring Hate Speech  CC-BY-4.0  (HF dataset card metadata)
#   DICES                  CC BY 4.0  (google-research-datasets/dices-dataset README)
#   Wikipedia Talk labels  CC0        (figshare article 4563973)
#   GoEmotions             Apache-2.0 per HF card; google-research repo states no data licence (AMBIGUOUS, recorded in LICENSES.md)
set -euo pipefail
cd "$(dirname "$0")/.."
get() { mkdir -p "data/$1"; [ -s "data/$1/$2" ] || curl -fsSL -o "data/$1/$2" "$3"; }
get mhs mhs.parquet https://huggingface.co/datasets/ucberkeley-dlab/measuring-hate-speech/resolve/main/data/train-00000-of-00001.parquet
get dices diverse_safety_adversarial_dialog_350.csv https://raw.githubusercontent.com/google-research-datasets/dices-dataset/main/350/diverse_safety_adversarial_dialog_350.csv
get dices diverse_safety_adversarial_dialog_990.csv https://raw.githubusercontent.com/google-research-datasets/dices-dataset/main/990/diverse_safety_adversarial_dialog_990.csv
for i in 1 2 3; do get goemotions goemotions_$i.csv https://storage.googleapis.com/gresearch/goemotions/data/full_dataset/goemotions_$i.csv; done
mkdir -p data/wikitalk
curl -fsSL -o data/wikitalk/figshare_4563973.json https://api.figshare.com/v2/articles/4563973
for f in toxicity_annotations.tsv toxicity_annotated_comments.tsv toxicity_worker_demographics.tsv; do
  url=$(python3 -c "import json,sys;print([x['download_url'] for x in json.load(open('data/wikitalk/figshare_4563973.json'))['files'] if x['name']==sys.argv[1]][0])" "$f")
  get wikitalk "$f" "$url"
done
curl -fsSL -o data/hatexplain_post_id_divisions.json https://raw.githubusercontent.com/hate-alert/HateXplain/master/Data/post_id_divisions.json
# Verify every file against the checksums committed with this repository, so an
# upstream change is caught before any analysis runs.
( cd data && shasum -a 256 -c ../scripts/data.sha256 )
echo "ok"; du -sh data/*
