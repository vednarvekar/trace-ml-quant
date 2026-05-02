# Trace ML Quant

Trace ML Quant is a quantitative trading research project focused on learning market structure from multi-timeframe data.

The long-term goal is to build a trading intelligence system that can combine:

1. price action understanding
2. engineered quantitative features
3. market context
4. risk management
5. execution logic

The current codebase is centered on the first serious milestone:

`training a multi-timeframe CNN on synchronized OHLCV windows`

## Vision

Most retail trading systems depend heavily on a small set of indicators and fixed rule sets. This project takes a different approach.

The idea is to treat market behavior as a structured pattern-recognition problem across multiple timeframes:

1. `1-minute` data captures microstructure, momentum bursts, and entry precision
2. `5-minute` data captures intraday structure and local trend behavior
3. `1-hour` data captures broader trend context and higher-level market direction

Instead of relying on a single timeframe or a few indicators, the objective is to train models that can learn how these layers interact.

## Current Scope

At the moment, the repo implements a CNN training pipeline for directional classification.

The current model answers a narrower question than the full project vision:

`Given recent synchronized 1m, 5m, and 1h OHLCV windows, will price likely move up, down, or sideways over the next short horizon?`

That means this is currently a research baseline for prediction, not yet a full live trading engine.

## Current Architecture

The active model is a multi-input CNN with three branches:

1. `Micro branch`
   - input: `1-minute` candles
   - purpose: immediate momentum and short-term structure
2. `Intraday branch`
   - input: `5-minute` candles
   - purpose: session-level trend and local support/resistance behavior
3. `Macro branch`
   - input: `1-hour` candles
   - purpose: broader directional context

Each branch processes its own window through convolution layers, then the learned representations are fused and passed through fully connected layers to classify:

1. `0 = Neutral`
2. `1 = Buy`
3. `2 = Sell`

Implementation:

1. model definition: [models/cnn/model.py](/home/vedvn/trace-ml-quant/models/cnn/model.py:1)
2. training script: [models/cnn/train.py](/home/vedvn/trace-ml-quant/models/cnn/train.py:1)

## Dataset Design

The dataset is built from synchronized OHLCV data across three timeframes:

1. `1min`
2. `5min`
3. `1hr`

Each training sample currently consists of:

1. last `60` candles of `1-minute` data
2. last `60` candles of `5-minute` data
3. last `60` candles of `1-hour` data

Each candle includes:

1. `open`
2. `high`
3. `low`
4. `close`
5. `volume`
6. `oi`

Each window is normalized locally before training.

### Labeling Logic

Labels are generated from future `5-minute` price movement:

1. `Buy`
   - future move greater than `+0.5%`
2. `Sell`
   - future move less than `-0.5%`
3. `Neutral`
   - anything in between

Current forecast horizon:

1. `10` future `5-minute` candles
2. approximately `50 minutes`

## Project Workflow

The current end-to-end pipeline is:

1. collect raw OHLCV data
2. organize it by timeframe in `data/raw`
3. create synchronized training windows for each stock
4. generate labels from future movement
5. save processed arrays per stock
6. merge all processed stocks into one master dataset
7. train the CNN on the merged dataset
8. save the trained weights

### Raw Data

Raw market data is stored in:

1. `data/raw/1min`
2. `data/raw/5min`
3. `data/raw/1hr`

The repo already contains raw data for:

1. `reliance`
2. `hdfc`
3. `tataMotors`
4. `nifty`

### Data Preparation

The preprocessing script:

1. loads OHLCV JSON for a stock across all three timeframes
2. aligns samples using the `5-minute` timestamp as the anchor
3. extracts rolling windows from all three timeframes
4. normalizes the windows
5. generates class labels
6. saves the processed arrays

Script:

1. [script/prepare_data.py](/home/vedvn/trace-ml-quant/script/prepare_data.py:1)

Run:

```bash
./.venv/bin/python script/prepare_data.py --stocks all
```

### Dataset Merge

Processed per-stock arrays are merged into a master training dataset.

Script:

1. [script/merge_data.py](/home/vedvn/trace-ml-quant/script/merge_data.py:1)

Run:

```bash
./.venv/bin/python script/merge_data.py
```

This creates:

1. `data/master_training/MASTER_X1.npy`
2. `data/master_training/MASTER_X5.npy`
3. `data/master_training/MASTER_XH.npy`
4. `data/master_training/MASTER_y.npy`

### Model Training

The training script:

1. loads the master arrays
2. adds a channel dimension for CNN input
3. builds the multi-branch CNN
4. uses weighted cross-entropy to handle class imbalance
5. trains with Adam, weight decay, gradient clipping, and LR scheduling
6. saves the trained model

Run:

```bash
cd models/cnn
../../.venv/bin/python train.py --epochs 30 --batch-size 64 --lr 5e-4 --device cpu
```

Output:

1. `models/pattern_master_cnn.pth`

## Current Strengths

The repo already has a solid starting structure for experimentation:

1. multi-timeframe dataset design
2. synchronized window generation
3. merged master training arrays
4. dedicated CNN architecture
5. class-weighted training setup

This is enough to begin serious baseline research.

## Current Limitations

The codebase is still in the baseline research phase.

Important gaps that remain:

1. no proper train/validation/test split yet
2. no time-based evaluation pipeline yet
3. no confusion matrix or class-wise evaluation metrics yet
4. no backtesting layer yet
5. no trading cost, slippage, or execution modeling yet
6. no risk engine yet
7. no external feature models yet

Because of this, model accuracy alone should not be treated as proof of profitability.

## Long-Term Roadmap

After the CNN baseline is stable, the broader project can expand into a layered quant system.

### Stage 1

Pattern recognition from raw multi-timeframe OHLCV using CNNs

### Stage 2

Engineered quant features such as:

1. EMA
2. MACD
3. ADX
4. RSI
5. ATR
6. VWAP
7. volatility features
8. candlestick pattern flags
9. market regime and sector context

### Stage 3

Decision and meta-modeling:

1. combine CNN output with engineered features
2. estimate trade confidence
3. learn when to trade and when to skip

### Stage 4

Risk and execution:

1. position sizing
2. stop-loss design
3. take-profit logic
4. drawdown control
5. live execution constraints

### Stage 5

Backtesting, paper trading, and only then live deployment

## Recommended Focus

The most important rule for this project is:

`do not optimize everything at once`

The correct current focus is:

1. make the CNN baseline reliable
2. validate it properly
3. test whether it produces actionable signal
4. only then expand into more models and features

That means the next high-value improvements are:

1. time-safe train/validation/test split
2. evaluation metrics beyond loss
3. experiment tracking
4. backtesting the model outputs as trades

## Repository Structure

```text
trace-ml-quant/
├── data/
│   ├── raw/
│   │   ├── 1min/
│   │   ├── 5min/
│   │   └── 1hr/
│   ├── processed/
│   └── master_training/
├── models/
│   ├── cnn/
│   │   ├── model.py
│   │   └── train.py
│   └── train/
├── script/
│   ├── prepare_data.py
│   ├── merge_data.py
│   └── caterpillar.ts
├── cnn_model.md
└── PROJECT_SUGGESTIONS.md
```

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

If using the local virtualenv already present in the repo, prefer:

```bash
./.venv/bin/pip install -r requirements.txt
```

## Quick Start

1. Prepare processed datasets:

```bash
./.venv/bin/python script/prepare_data.py --stocks all
```

2. Merge master training arrays:

```bash
./.venv/bin/python script/merge_data.py
```

3. Train the CNN:

```bash
cd models/cnn
../../.venv/bin/python train.py --epochs 30 --batch-size 64 --lr 5e-4 --device cpu
```

## Notes

This README now reflects the real state of the repo:

1. the vision is broad
2. the implementation is currently focused on CNN-based market pattern learning
3. the project is still in the research and validation stage

That is the right place to be. A strong baseline matters more than a large but unverified system.
