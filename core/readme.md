## RewardModel Inference Readme

### Environment

**港科机器上：**
conda activate new_rm
我的codebase在 `/data/yrb/musicarena/Haiwen/offline_data/cmi-arena/RewardModel/`,可以参考


**自己配环境：**
```bash
conda create -n muqrm python=3.10 -y
conda activate muqrm
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install flash-attn --no-build-isolation
pip install -r reward_model/requirements.txt
pip install -e .
```

## Quick Start

```python
from inference import RewardModelInference

# Load the model
model = RewardModelInference(
    checkpoint="path/to/checkpoint",  # File or directory
    device="cuda:0"
)
# 在港科机器上

model = RewardModelInference(
    checkpoint="/data/yrb/musicarena/Haiwen/offline_data/cmi-arena/RewardModel/1118_muqmulan",  # /data/yrb/musicarena/Haiwen/offline_data/cmi-arena/RewardModel/1118_muqmulan/ckpt/full_model.pt 也可
    device="cuda:0"
)

# Score a single audio file
scores = model.score(
    audio="song.mp3",
    text="A cheerful pop song with upbeat rhythm"
)

print(f"Alignment: {scores['alignment']:.3f}")  # Text-music alignment
print(f"Quality: {scores['quality']:.3f}")       # Music quality
```

## API Reference

### `RewardModelInference`

Main class for music scoring.

#### Constructor

```python
RewardModelInference(
    checkpoint: str,      # Path to checkpoint file or directory
    device: str = "cuda:0",  # Device ("cuda:0", "cpu", etc.)
    sr: int = 24000       # Audio sample rate
)
```

**Checkpoint formats supported:**
- Single file: `path/to/model.pt`
- Directory: `path/to/dir/` containing:
  - `ckpt/full_model.pt` (or `reward_model.*.pt`)
  - `config.yaml`

#### Methods

##### `score()` - Score single audio

```python
scores = model.score(
    audio: str | Tensor,      # Audio file path or waveform tensor
    text: str = "",           # Text prompt
    lyrics: str = "",         # Lyrics (optional)
    ref_audio: str = None,    # Reference audio (optional)
    max_dur: float = 30.0,    # Max duration per interval
    dur_step: float = None    # Sliding window step (None = no sliding)
) -> Dict[str, float]         # {'alignment': float, 'quality': float}
```

##### `score_batch()` - Score multiple audio files

```python
scores = model.score_batch(
    inputs: List[Dict],       # List of input dicts
    batch_size: int = 4,
    max_dur: float = 30.0,
    dur_step: float = None,
    show_progress: bool = False
) -> np.ndarray               # [N, 2] array, [:, 0]=alignment, [:, 1]=quality
```

**Input dict format:**
```python
{
    "audio": "path/to/audio.mp3",  # Required
    "text": "description",          # Optional, default ""
    "lyrics": "song lyrics",        # Optional, default ""
    "ref_audio": "reference.mp3"    # Optional
}
```

## Examples

### Basic Scoring

```python
from inference import RewardModelInference

model = RewardModelInference("checkpoint/")

# Simple scoring
scores = model.score("song.mp3", text="An energetic rock song")
print(f"Alignment: {scores['alignment']:.3f}, Quality: {scores['quality']:.3f}")
```

### Scoring with Lyrics

```python
scores = model.score(
    audio="ballad.mp3",
    text="A romantic love ballad",
    lyrics="I love you more than words can say..."
)
```

### Scoring with Reference Audio

```python
# Useful for style transfer evaluation
scores = model.score(
    audio="generated.mp3",
    text="Jazz piano piece",
    ref_audio="reference_jazz.mp3"
)
```

### Batch Scoring

```python
inputs = [
    {"audio": "song1.mp3", "text": "Pop song"},
    {"audio": "song2.mp3", "text": "Classical piece"},
    {"audio": "song3.mp3", "text": "Electronic beat"},
]

scores = model.score_batch(inputs, batch_size=8) 

# 格式： [np.ndarray] shape (N, 2), where [:,0]=alignment, [:,1]=quality
for i, inp in enumerate(inputs):
    print(f"{inp['audio']}: alignment={scores[i,0]:.3f}, quality={scores[i,1]:.3f}")
```

### Scoring Long Audio with Sliding Window

For audio longer than `max_dur`, use sliding window to get averaged scores:

```python
# Score 5-minute song with 30s windows, 10s step
scores = model.score(
    audio="long_song.mp3",
    text="Epic orchestral soundtrack",
    max_dur=30.0,
    dur_step=10.0  # Creates overlapping 30s windows every 10s
)

scores:
{
    "alignment": xxx,
    "quality": xxx
}
```



### Using Pre-loaded Waveform

```python
import torchaudio

# Load audio yourself
waveform, sr = torchaudio.load("song.mp3")
waveform = waveform.mean(0)  # Mono

# Resample to 24kHz if needed
if sr != 24000:
    waveform = torchaudio.transforms.Resample(sr, 24000)(waveform)

# Score
scores = model.score(audio=waveform, text="Description")
```


### Sample Benchmarking Script
`` provides a full example of loading a dataset and evaluating the model on multiple samples, saving results to JSONL. See the script for details on usage and options.