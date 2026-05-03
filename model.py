# model.py
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

class Model:
    def __init__(self, k=3):
        self.scaler = StandardScaler()
        self.nn = NearestNeighbors(n_neighbors=k, metric='euclidean')
        self.file_ids = None
        self.k = k

    def fit(self, features, file_ids):
        scaled = self.scaler.fit_transform(features)
        self.nn.fit(scaled)
        self.file_ids = np.array(file_ids)

    def find_similar(self, new_features, k=None):
        k = k or self.k
        scaled = self.scaler.transform(new_features.reshape(1, -1))
        distances, indices = self.nn.kneighbors(scaled, n_neighbors=k)

        return [
            (self.file_ids[idx], float(dist))
            for idx, dist in zip(indices[0], distances[0])
        ]