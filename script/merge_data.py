import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data/processed"
MASTER_DIR = BASE_DIR / "data/master_training"
MASTER_DIR.mkdir(parents=True, exist_ok=True)

STOCKS = ["reliance", "hdfc", "tataMotors", "nifty"]

def get_total_samples():
    total = 0
    for stock in STOCKS:
        y = np.load(PROCESSED_DIR / stock / "y_labels.npy")
        total += len(y)
    return total

def merge_memory_efficient():
    total_samples = get_total_samples()
    print(f"Total samples to merge: {total_samples}")

    # 1. Create empty 'Memory Mapped' files on disk (They don't use RAM)
    # Shape: (Total, 60, 6)
    m1 = np.memmap(MASTER_DIR / 'X1.tmp', dtype='float32', mode='w+', shape=(total_samples, 60, 6))
    m5 = np.memmap(MASTER_DIR / 'X5.tmp', dtype='float32', mode='w+', shape=(total_samples, 60, 6))
    mH = np.memmap(MASTER_DIR / 'XH.tmp', dtype='float32', mode='w+', shape=(total_samples, 60, 6))
    mY = np.memmap(MASTER_DIR / 'Y.tmp',  dtype='int64',   mode='w+', shape=(total_samples,))

    cursor = 0
    for stock in STOCKS:
        print(f"Streaming {stock} to disk...")
        x1 = np.load(PROCESSED_DIR / stock / "X_1min.npy")
        x5 = np.load(PROCESSED_DIR / stock / "X_5min.npy")
        xh = np.load(PROCESSED_DIR / stock / "X_1hr.npy")
        y  = np.load(PROCESSED_DIR / stock / "y_labels.npy")
        
        num = len(y)
        m1[cursor:cursor+num] = x1
        m5[cursor:cursor+num] = x5
        mH[cursor:cursor+num] = xh
        mY[cursor:cursor+num] = y
        cursor += num
        
        # Clean up RAM after each stock
        del x1, x5, xh, y 

    print("Flushing data to SSD...")
    m1.flush(); m5.flush(); mH.flush(); mY.flush()
    
    # 2. Shuffle indices (this fits in RAM easily)
    print("Generating shuffled indices...")
    idx = np.arange(total_samples)
    np.random.shuffle(idx)

    # 3. Save as final .npy files
    print("Saving final Master files...")
    np.save(MASTER_DIR / "MASTER_X1.npy", m1[idx])
    np.save(MASTER_DIR / "MASTER_X5.npy", m5[idx])
    np.save(MASTER_DIR / "MASTER_XH.npy", mH[idx])
    np.save(MASTER_DIR / "MASTER_y.npy", mY[idx])

    print("Success! Data merged without crashing.")

if __name__ == "__main__":
    merge_memory_efficient()