# SoundMatch

A music similarity search system that uses audio feature extraction and a K-Nearest Neighbors model to find songs with similar acoustic characteristics. Given any MP3 file, SoundMatch returns the most acoustically similar tracks from the FMA dataset. > See [`SoundMatch_Project_Documentation.pdf`](SoundMatch_Project_Documentation.pdf) for a more detailed project write-up. This project was inspired by (https://www.youtube.com/watch?v=V5Psciq3T_g&t=331s).

---

## How It Works

1. **Feature Extraction** — Each MP3 is analyzed and 156 acoustic features are extracted per song:
   - MFCC (Mel-Frequency Cepstral Coefficients): 13 means + 13 standard deviations
   - Mel Spectrogram: 64 means + 64 standard deviations
   - Tempo (BPM)
   - Onset Density (note onsets per second)
2. **Model Training** — Features are standardized and fed into a K-Nearest Neighbors model (k=3, Euclidean distance).
3. **Similarity Search** — A query audio file has its features extracted and the k=3 most similar songs from the dataset are returned.

---

## Project Structure

```
SoundMatch/
├── main.py               # Entry point — orchestrates the full pipeline
├── model.py              # K-NN model wrapper (fit + find_similar)
├── data.py               # Audio feature extraction and CSV handling
├── save_load_model.py    # Model persistence (joblib save/load)
├── requirements.txt      # Python dependencies
├── prepared_data.csv     # Pre-computed features (generated on first run)
├── soundmatch.joblib     # Trained model (generated on first run)
├── test.mp3              # Your query audio file (place here)
└── data_set/             # Dataset folder — all MP3s go directly here
    ├── 000002.mp3
    ├── 000005.mp3
    ├── ...
    └── 155066.mp3
```

---

## Dataset

The project uses the **Free Music Archive (FMA)** dataset — specifically the `fma_small` split (~8,000 MP3 files at 30 seconds each).

**The dataset is not included in this repository** due to its size. Download it from the official source:

> **FMA GitHub:** [https://github.com/mdeff/fma](https://github.com/mdeff/fma)

Download the `fma_small.zip` file from the releases section of that repository.

### IMPORTANT — File Structure After Download

After extracting, the FMA archive places MP3 files in **numbered subfolders** (e.g. `fma_small/000/`, `fma_small/001/`, etc.). **This will not work with SoundMatch.** All MP3 files must be in a **single flat folder** with no subfolders.

You need to move all MP3 files out of the subfolders and into one folder named `fma_small/` directly inside the project root. Every `.mp3` file must sit at `fma_small/<filename>.mp3` — not deeper.

**On Windows (PowerShell), you can flatten the directory like this:**

```powershell
# Run this from the SoundMatch project root
Get-ChildItem -Path "fma_small" -Recurse -Filter "*.mp3" | Move-Item -Destination "fma_small\"
Get-ChildItem -Path "fma_small" -Directory | Remove-Item -Recurse
```

After running this, your `fma_small/` folder should contain ~8,000 `.mp3` files directly, with no subdirectories.

---

## Setup

### 1. Clone or download the repository

```powershell
git clone <repo-url>
cd SoundMatch
```

### 2. Create and activate a virtual environment

```powershell
python -m venv venv

# Activate the virtual environment
venv/Scripts/Activate
```

### 3. Install dependencies

```powershell
pip install -r requirements.txt
```

---

## Running SoundMatch

Place the MP3 file you want to query (e.g. `test.mp3`) in the project root, then run:

```powershell
python main.py
```

You will be prompted:

```
Enter path to test audio file (e.g. test.mp3):
```

Press **Enter** to use the default `test.mp3`, or type a path to any MP3 file.

### First Run

On the first run (when no model exists yet), SoundMatch will:

1. Process all MP3 files in `fma_small/` and extract features → writes `prepared_data.csv`
2. Train the K-NN model on those features → writes `soundmatch.joblib`
3. Run the similarity search on your query file

This can take a while depending on your hardware (~8,000 songs to process). Subsequent runs skip steps 1 and 2 and go straight to the search.

### Output

```
Similar songs:
  000123.mp3  (distance: 12.453)
  004521.mp3  (distance: 13.871)
  001099.mp3  (distance: 14.220)
```

The lower the distance, the more acoustically similar the song.

---

## Requirements

- Python 3.10+
- All dependencies listed in `requirements.txt`

Key libraries:
| Library | Purpose |
|---|---|
| `librosa` | Audio analysis and feature extraction |
| `scikit-learn` | K-Nearest Neighbors model |
| `numpy` | Numerical computing |
| `scipy` | Scientific computing |
| `joblib` | Model serialization |

---

## Notes

- Files shorter than 1 second are skipped during feature extraction.
- Corrupted or unreadable MP3 files are skipped with an error message — they will not stop the process.
- All audio is resampled internally to 22,050 Hz regardless of the source file's sample rate.
- `prepared_data.csv` and `soundmatch.joblib` are generated automatically and can be deleted to force a full retrain.

---

## Limitations

- Hand-crafted features capture surface acoustic properties (timbre, rhythm) but not deeper musical meaning. Two songs that feel similar to a human listener may not be close in this feature space.
- Distances are unitless — only relative ranking is meaningful.
- Long input songs (≥30 sec) are processed in full, but FMA training tracks are uniformly 30 seconds, so feature comparability decreases for very long inputs.
- A learned audio embedding (e.g., CLAP, OpenL3) would likely produce better-quality similarity at the cost of complexity.

---

## AI Disclosure

Claude (Anthropic) was used to assist with explanations, code review, and documentation polish. All design, architecture, and implementation decisions were made and verified by the author.

---

## License

This project is licensed under the MIT License.

> © **2025 Neal**  

> *Feel free to modify, improve, and share — just include credit to the original author.*
