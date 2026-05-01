import pandas as pd
import numpy as np
import json
from pathlib import Path

WINDOW = 60
HORIZON = 10

BASE_DIR = Path(__file__).resolve().parent.parent
data_path = BASE_DIR / "data" / "raw/reliance_5min_ohlcv.json"
save_path_X = BASE_DIR / "data" / "processed/reliance-X_train.npy"
save_path_y = BASE_DIR / "data" / "processed/reliance-y_train.npy"

def process_to_3d(filename):
    with open(filename, 'r') as f:
        df = pd.DataFrame(json.load(f))

    features = df[['open', 'high', 'low', 'close', 'volume']].values

    X, y = [], []

    for i in range(len(features) - WINDOW - HORIZON):
        # 1. Grabbing Window (60, 5)
        snippet = features[i: i + WINDOW]

        # 2. Normalize (Scale 0 to 1)
        snippet_min = snippet.min(axis=0)
        snippet_max = snippet.max(axis=0)
        norm_snippet = (snippet - snippet_min) / (snippet_max - snippet_min + 1e-7)

        # 3. Create the Answer (y)
        price_now = features[i + WINDOW - 1][3]
        price_future = features[i + WINDOW + HORIZON - 1][3]

        # Did it go up > 0.5%?
        change = (price_future - price_now) / price_now
        lable = 1 if change > 0.005 else 0

        X.append(norm_snippet)
        y.append(lable)

    return np.array(X), np.array(y)
    
X_train, y_train = process_to_3d(data_path)

np.save(save_path_X, X_train)
np.save(save_path_y, y_train)

print(f"Success! Data Shape: {X_train.shape}")
print(f"Positive samples found: {np.sum(y_train)} out of {len(y_train)}")