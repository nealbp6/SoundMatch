# train.py
from model import Model
from save_load_model import save_model, load_model
from data import load_from_csv, prepare_single_file, prepared_data
from pathlib import Path

def train_and_save_model():
    features, file_ids = load_from_csv(csv_file)

    model = Model(k=3)
    model.fit(features, file_ids)
    save_model(model, model_name)

    print(f"Trained on {len(features)} songs and saved to {model_name}")

def find_similar_songs(new_audio_file):
    model = load_model(model_name)

    # Extract features from new_audio_file (not shown here, but similar to data.py)
    new_features = prepare_single_file(new_audio_file)

    similar_songs = model.find_similar(new_features)
    print(f"Similar songs to {new_audio_file}:")
    for file_id, dist in similar_songs:
        print(f"{file_id} (distance: {dist:.2f})")

audio_dir = 'data_set'

csv_file = 'prepared_data.csv'
model_name = 'soundmatch.joblib'
test_audio_file = input("Enter path to test audio file (e.g. test.mp3): ")
if not test_audio_file:
    test_audio_file = 'test.mp3'

if not Path(model_name).exists(): # check if mdel exists, if not create it
    # create csv
    prepared_data(audio_dir, sr=22050)

    # train model and save
    train_and_save_model()

# test similarity search
find_similar_songs(test_audio_file)