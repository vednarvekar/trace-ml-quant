import pandas as pd
import numpy as np
import json
from pathlib import Path

# --- CONFIGURATION ---
WINDOW = 60      # We keep 60 candles for each "eye"
HORIZON_MINS = 10 # Looking 10 candles (50 mins) into the future

# Candle counts for history
COUNT_1M = 60  # 1 hour of micro detail
COUNT_5M = 60  # 5 hours of intraday patterns
COUNT_1H = 60  # 60 hours (2.5 days) of macro trend/S&R

BASE_DIR = Path(__file__).resolve().parent.parent
paths = {
    "1min": BASE_DIR / "data/raw/1min/reliance_ohlcv.json",
    "5min": BASE_DIR / "data/raw/5min/reliance_ohlcv.json",
    "1hr":  BASE_DIR / "data/raw/1hr/reliance_ohlcv.json"
}

# Ensure save directory exists
save_dir = BASE_DIR / "data/processed/reliance"
save_dir.mkdir(parents=True, exist_ok=True)

def load_and_prep(path):
    with open(path, 'r') as f:
        df = pd.DataFrame(json.load(f))
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    # Drop duplicates just in case API double-sent data
    return df.drop_duplicates('timestamp').set_index('timestamp').sort_index()

def process_master_chunks():
    print("--- Loading Dataframes ---")
    df1 = load_and_prep(paths["1min"])
    df5 = load_and_prep(paths["5min"])
    dfH = load_and_prep(paths["1hr"])

    X1, X5, XH, y = [], [], [], []

    print("--- Syncing and Processing Windows ---")
    # Start after 60 hours (approx 720 5-min candles) to ensure 1hr history exists
    start_index = 720 
    
    for i in range(start_index, len(df5) - HORIZON_MINS):
        anchor_time = df5.index[i]

        # Grab exact history counts ending at anchor_time
        snip1 = df1[:anchor_time].tail(COUNT_1M).values
        snip5 = df5[:anchor_time].tail(COUNT_5M).values
        snipH = dfH[:anchor_time].tail(COUNT_1H).values

        # VERIFICATION: Ensure no gaps in history
        if len(snip1) < COUNT_1M or len(snip5) < COUNT_5M or len(snipH) < COUNT_1H:
            continue

        # NORMALIZATION: Local scaling for pattern recognition
        def norm(s): 
            return (s - s.min(axis=0)) / (s.max(axis=0) - s.min(axis=0) + 1e-7)

        # LABELING: 0.5% Threshold for Buy/Sell
        price_now = df5.iloc[i]['close']
        price_future = df5.iloc[i + HORIZON_MINS]['close']
        change = (price_future - price_now) / price_now

        if change > 0.005:
            label = 1    # BUY
        elif change < -0.005:
            label = 2    # SELL
        else:
            label = 0    # NEUTRAL

        X1.append(norm(snip1))
        X5.append(norm(snip5))
        XH.append(norm(snipH))
        y.append(label)

        if len(y) % 1000 == 0:
            print(f"Processed {len(y)} samples...")

    return np.array(X1), np.array(X5), np.array(XH), np.array(y)

# Execution
X1_final, X5_final, XH_final, y_final = process_master_chunks()

# SAVING
np.save(save_dir / "X_1min.npy", X1_final)
np.save(save_dir / "X_5min.npy", X5_final)
np.save(save_dir / "X_1hr.npy", XH_final)
np.save(save_dir / "y_labels.npy", y_final)

# --- FINAL ANALYSIS ---
print("\n--- DATA PROCESSING COMPLETE ---")
print(f"Total Samples Generated: {len(y_final)}")
print(f"Micro Input Shape (1min): {X1_final.shape}")
print(f"Intraday Input Shape (5min): {X5_final.shape}")
print(f"Macro Input Shape (1hr): {XH_final.shape}")
print("-" * 30)
print(f"Buy (1) Signals: {np.sum(y_final == 1)} ({np.sum(y_final == 1)/len(y_final)*100:.2f}%)")
print(f"Short (2) Signals: {np.sum(y_final == 2)} ({np.sum(y_final == 2)/len(y_final)*100:.2f}%)")
print(f"Neutral (0) Signals: {np.sum(y_final == 0)} ({np.sum(y_final == 0)/len(y_final)*100:.2f}%)")
print("-" * 30)
print("The data is now ready for a 3-Input Functional CNN.")