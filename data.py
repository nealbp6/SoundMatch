# data.py

# prepare data, sound intensity, sound level, sequentually
import numpy as np
import librosa
from pathlib import Path
import csv

def prepared_data(audio_dir, sr=22050):  # to csv due to memory issues
    audio_files = sorted(Path(audio_dir).glob('*.mp3'))
    file_amount = len(audio_files)

    with open('prepared_data.csv', 'a', newline='') as f:
        writer = csv.writer(f)

        for i, file in enumerate(audio_files, 1):
            try:
                audio, _ = librosa.load(file, sr=sr)  # audio also y

                # Skip files shorter than 1 second
                if len(audio) < sr:
                    print(f"Skipped {file.name}: too short")
                    continue

                # MFCC
                mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
                mfcc_mean = mfcc.mean(axis=1)
                mfcc_std = mfcc.std(axis=1)

                # Mel spectrogram (log-scaled)
                mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=64)
                mel_db = librosa.power_to_db(mel)
                mel_mean = mel_db.mean(axis=1)
                mel_std = mel_db.std(axis=1)

                # Tempo
                tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
                tempo = float(np.atleast_1d(tempo)[0])

                # Onset density per second
                onset_frames = librosa.onset.onset_detect(y=audio, sr=sr)
                onset_density = len(onset_frames) / (len(audio) / sr)

                # Build row: file_id first (string), then features (floats)
                row = [file.stem]
                row.extend(mfcc_mean.tolist())
                row.extend(mfcc_std.tolist())
                row.extend(mel_mean.tolist())
                row.extend(mel_std.tolist())
                row.extend([tempo, onset_density])

                writer.writerow(row)

                # Flush every 100 files so a crash doesn't lose everything
                if i % 100 == 0:
                    f.flush()

                print(f"Processed {file.name} ({i}/{file_amount})")

            except Exception as e:
                print(f"Failed on {file.name}: {e}")
                continue
    
def prepare_single_file(audio_file, sr=22050):
    audio, _ = librosa.load(audio_file, sr=sr)

    if len(audio) < sr:
        raise ValueError("Audio too short")

    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    mfcc_mean = mfcc.mean(axis=1)
    mfcc_std = mfcc.std(axis=1)

    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=64)
    mel_db = librosa.power_to_db(mel)
    mel_mean = mel_db.mean(axis=1)
    mel_std = mel_db.std(axis=1)

    tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
    tempo = float(np.atleast_1d(tempo)[0])

    onset_frames = librosa.onset.onset_detect(y=audio, sr=sr)
    onset_density = len(onset_frames) / (len(audio) / sr)

    features = np.concatenate([mfcc_mean, mfcc_std, mel_mean, mel_std, [tempo, onset_density]])
    return features


def load_from_csv(csv_file):
    features = []
    file_ids = []
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            file_ids.append(row[0])  # first column is file_id
            features.append(list(map(float, row[1:])))  # rest are features

            if len(features) % 100 == 0:
                print(f"Loaded {len(features)} rows")
    return np.array(features), file_ids
