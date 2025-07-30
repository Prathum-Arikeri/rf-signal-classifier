import os
import pickle
import numpy as np


def load_data(pkl_path):
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    return data


def preprocess_data(data):
    # Example preprocessing:
    # - Extract features and labels from loaded dict
    # - Normalize or reshape as needed
    # - Convert labels to integer indices

    X = data['X']  # assuming 'X' key contains signal samples
    y = data['Y']  # assuming 'Y' key contains labels

    # Normalize signals (example)
    X_norm = X / np.max(np.abs(X), axis=1, keepdims=True)

    return X_norm, y


def save_processed_data(X, y, output_dir='data/processed'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    np.save(os.path.join(output_dir, 'X.npy'), X)
    np.save(os.path.join(output_dir, 'y.npy'), y)
    print(f"Saved processed data to {output_dir}")


if __name__ == "__main__":
    raw_pkl = 'data/raw/RadioML2016.10a/RadioML2016.10a.pkl'
    data = load_data(raw_pkl)
    X, y = preprocess_data(data)
    save_processed_data(X, y)
