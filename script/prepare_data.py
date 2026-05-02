import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

# --- CONFIGURATION ---
HORIZON_5M_CANDLES = 10
COUNT_1M = 60
COUNT_5M = 60
COUNT_1H = 60
BUY_THRESHOLD = 0.005
SELL_THRESHOLD = -0.005

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
TIMEFRAME_DIRS = {"1min": "1min", "5min": "5min", "1hr": "1hr"}


def load_and_prep(path: Path) -> pd.DataFrame:
    with open(path, "r") as f:
        df = pd.DataFrame(json.load(f))

    if df.empty:
        raise ValueError(f"No rows found in {path}")

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.drop_duplicates("timestamp").set_index("timestamp").sort_index()

    required_cols = ["open", "high", "low", "close", "volume", "oi"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in {path}: {missing_cols}")

    return df[required_cols]


def available_stocks() -> list[str]:
    stocks = set()
    for timeframe in TIMEFRAME_DIRS.values():
        for path in (RAW_DIR / timeframe).glob("*_ohlcv.json"):
            stocks.add(path.name.replace("_ohlcv.json", ""))
    return sorted(stocks)


def resolve_stocks(requested: list[str]) -> list[str]:
    all_stocks = available_stocks()
    if not requested or requested == ["all"]:
        return all_stocks

    missing = [stock for stock in requested if stock not in all_stocks]
    if missing:
        raise ValueError(f"Unknown stocks: {missing}. Available: {all_stocks}")

    return requested


def paths_for_stock(stock: str) -> dict[str, Path]:
    return {
        "1min": RAW_DIR / "1min" / f"{stock}_ohlcv.json",
        "5min": RAW_DIR / "5min" / f"{stock}_ohlcv.json",
        "1hr": RAW_DIR / "1hr" / f"{stock}_ohlcv.json",
    }


def normalize_window(window: np.ndarray) -> np.ndarray:
    return (window - window.min(axis=0)) / (window.max(axis=0) - window.min(axis=0) + 1e-7)


def label_from_future_move(price_now: float, price_future: float) -> int:
    change = (price_future - price_now) / price_now
    if change > BUY_THRESHOLD:
        return 1
    if change < SELL_THRESHOLD:
        return 2
    return 0


def process_stock(stock: str) -> dict[str, object]:
    print(f"\n--- Processing {stock} ---")
    stock_paths = paths_for_stock(stock)

    for timeframe, path in stock_paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing {timeframe} file for {stock}: {path}")

    df1 = load_and_prep(stock_paths["1min"])
    df5 = load_and_prep(stock_paths["5min"])
    dfh = load_and_prep(stock_paths["1hr"])

    x1_list, x5_list, xh_list, y_list = [], [], [], []
    start_index = 720

    print("Syncing timeframe windows...")
    for i in range(start_index, len(df5) - HORIZON_5M_CANDLES):
        anchor_time = df5.index[i]

        snip1 = df1[:anchor_time].tail(COUNT_1M).values
        snip5 = df5[:anchor_time].tail(COUNT_5M).values
        sniph = dfh[:anchor_time].tail(COUNT_1H).values

        if len(snip1) < COUNT_1M or len(snip5) < COUNT_5M or len(sniph) < COUNT_1H:
            continue

        price_now = float(df5.iloc[i]["close"])
        price_future = float(df5.iloc[i + HORIZON_5M_CANDLES]["close"])
        label = label_from_future_move(price_now, price_future)

        x1_list.append(normalize_window(snip1))
        x5_list.append(normalize_window(snip5))
        xh_list.append(normalize_window(sniph))
        y_list.append(label)

        if len(y_list) % 10000 == 0:
            print(f"Processed {len(y_list)} samples for {stock}...")

    x1 = np.asarray(x1_list, dtype=np.float32)
    x5 = np.asarray(x5_list, dtype=np.float32)
    xh = np.asarray(xh_list, dtype=np.float32)
    y = np.asarray(y_list, dtype=np.int64)

    stock_output_dir = PROCESSED_DIR / stock
    stock_output_dir.mkdir(parents=True, exist_ok=True)

    np.save(stock_output_dir / "X_1min.npy", x1)
    np.save(stock_output_dir / "X_5min.npy", x5)
    np.save(stock_output_dir / "X_1hr.npy", xh)
    np.save(stock_output_dir / "y_labels.npy", y)

    label_counts = {int(label): int((y == label).sum()) for label in [0, 1, 2]}
    summary = {
        "stock": stock,
        "samples": int(len(y)),
        "x_1min_shape": list(x1.shape),
        "x_5min_shape": list(x5.shape),
        "x_1hr_shape": list(xh.shape),
        "label_counts": label_counts,
        "config": {
            "count_1m": COUNT_1M,
            "count_5m": COUNT_5M,
            "count_1h": COUNT_1H,
            "horizon_5m_candles": HORIZON_5M_CANDLES,
            "buy_threshold": BUY_THRESHOLD,
            "sell_threshold": SELL_THRESHOLD,
        },
    }

    with open(stock_output_dir / "dataset_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved processed arrays for {stock} to {stock_output_dir}")
    print(f"Samples: {summary['samples']} | Labels: {label_counts}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare synchronized CNN datasets from raw OHLCV JSON.")
    parser.add_argument(
        "--stocks",
        nargs="+",
        default=["all"],
        help="Stock names to process, or 'all' to process every available stock.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stocks = resolve_stocks(args.stocks)
    summaries = [process_stock(stock) for stock in stocks]

    print("\n--- DATA PROCESSING COMPLETE ---")
    for summary in summaries:
        samples = summary["samples"]
        labels = summary["label_counts"]
        print(f"{summary['stock']}: {samples} samples | labels {labels}")


if __name__ == "__main__":
    main()
