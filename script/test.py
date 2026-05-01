import pandas as pd
from pathlib import Path
import numpy as np
import json

# Load your data
BASE_DIR = Path(__file__).resolve().parent.parent
data_path = BASE_DIR / "data" / "raw/hdfc_5min_ohlcv.json"

# with open(data_path, 'r') as f:
#     df = pd.DataFrame(json.load(f))
#     print("File loaded successfully!")

# # Calculate the % change over every 10-candle period
# df['future_change'] = df['close'].shift(-10) / df['close'] - 1

# # Drop the last 10 rows (where we don't have future data)
# changes = df['future_change'].dropna()

# print(f"Average move in 50 mins: {changes.mean()*100:.10f}%")
# print(f"Standard Deviation: {changes.std()*100:.2f}%")

# # Check how many samples we get at different targets
# for target in [0.002, 0.005, 0.01, 0.02]:
#     count = (changes > target).sum()
#     percent = (count / len(changes)) * 100
#     print(f"Target {target*100}%: {count} samples ({percent:.1f}% of data)")


# Checking Data
data_path_X = BASE_DIR / "data" / "processed/tataMotors-X_train.npy"
data_path_y = BASE_DIR / "data" / "processed/tataMotors-y_train.npy"

X = np.load(data_path_X)
y = np.load(data_path_y)

print("For Tata Motors: ")
print(f"Total Windows: {len(y)}")
print(f"Buy Signals (1): {np.sum(y)}")
print(f"No-Trade Signals (0): {len(y) - np.sum(y)}")
print(f"Success Rate in Data: {(np.sum(y)/len(y))*100:.2f}%")