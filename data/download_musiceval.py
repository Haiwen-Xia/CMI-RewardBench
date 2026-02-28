import soundfile as sf
import os

def load_MusicEval(PATH="", split="test"):
    from datasets import load_dataset


    print(f"Loading MusicEval dataset ({split} split)...")
    dataset = load_dataset(
        "parquet",
        data_files={
            "test": "test-00000-of-00001.parquet",
        },
        data_dir=f"{PATH}dataset/MusicEval_data/data",
    )

    output_audio_dir = os.path.join(PATH, f"dataset/MusicEval_data/{split}_audio")
    os.makedirs(output_audio_dir, exist_ok=True)

    for idx, data in enumerate(dataset[split]):
        audio = data["audio"]["array"]
        sample_rate = data["audio"]["sampling_rate"]
        file_path = f"data/MusicEval_data/{split}_audio/{idx}.wav"
        if not os.path.exists(file_path):
            sf.write(file_path, audio, sample_rate)