# save_load_model.py
import joblib

def save_model(model, filename="model.joblib"):
    joblib.dump(model, filename)

def load_model(filename="model.joblib"):
    return joblib.load(filename)