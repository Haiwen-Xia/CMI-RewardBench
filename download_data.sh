#!/usr/bin/env bash
data_root='data'
pip install huggingface_hub
pip install datasets
pip install soundfile

echo "needs to login to huggingface hub"
read -r -p "Logged in..."

mkdir -p "$data_root"
echo "Downloading MusicEval dataset..."

echo "Downloading MusicEval dataset..."
python "$data_root"/download_musiceval.py


echo "Downloading PAM Music..."
wget -c -L --content-disposition \
  "https://zenodo.org/records/10737388/files/human_eval.zip?download=1"
unzip -q human_eval.zip -d "$data_root"
rm human_eval.zip

echo "Downloading MusicArena dataset..."
hf download music-arena/music-arena-dataset --local-dir "$data_root/MusicArena_data" --repo-type dataset

echo "Downloading CMI-Pref dataset..."
hf download HaiwenXia/cmi-pref --local-dir "$data_root/cmi-pref" --repo-type dataset
