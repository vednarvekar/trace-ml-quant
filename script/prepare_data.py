import pandas as pd
import numpy as np
import json

WINDOW = 60
HORIZON = 10

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
        lable = 1 if change > 0.05 else 0

        X.append(norm_snippet)
        y.append(lable)

    return np.array(X), np.array(y)
    
X_train, y_train = process_to_3d('../raw/reliance_5min_ohlcv.json')

np.save('../data/processed/reliance-X_train.npy', X_train)
np.save('../data/processed/reliance-y_train.npy', y_train)

print(f"Success! Data Shape: {X_train.shape}")
print(f"Positive samples found: {np.sum(y_train)} out of {len(y_train)}")