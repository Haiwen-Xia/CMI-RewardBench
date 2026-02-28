#!/usr/bin/env bash

echo "needs to login to huggingface hub"
read -r -p "Logged in..."

mkdir -p "data"
echo "Downloading MusicEval dataset..."

echo "Downloading MusicEval dataset..."

hf download BAAI/MusicEval data/test-00000-of-00001.parquet --local-dir data/MusicEval_data --repo-type dataset 

python "data"/download_musiceval.py

echo "Downloaded MusicEval dataset to data/MusicEval_data/"

echo "Downloading PAM Music..."
wget -c -L --content-disposition "https://zenodo.org/records/10737388/files/human_eval.zip?download=1"
unzip -q human_eval.zip -d "data"
rm human_eval.zip

echo "Downloaded PAM Music dataset to data/human_eval/"

echo "Downloading MusicArena dataset..."
huggingface-cli download music-arena/music-arena-dataset --local-dir "data/MusicArena_data" --repo-type dataset  --resume-download

echo "Downloaded MusicArena dataset to data/MusicArena_data/"

echo "Downloading CMI-Pref dataset..."
huggingface-cli download HaiwenXia/cmi-pref --local-dir data/cmi-pref --repo-type dataset  --resume-download

echo "Downloaded CMI-Pref dataset to data/cmi-pref/"