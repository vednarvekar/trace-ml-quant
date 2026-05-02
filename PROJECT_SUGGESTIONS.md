# Trace ML Quant: Vision, Workflow, and Focus

## 1. What this project is trying to become

This project is aiming to become a multi-stage trading intelligence system, not just a chart-pattern classifier.

The strongest version of the idea is:

1. A pattern model understands short-term, intraday, and higher-timeframe price structure.
2. A feature model understands trend, momentum, volatility, volume, options context, and market regime.
3. A decision layer converts predictions into actual trade actions.
4. A risk layer decides position size, stop loss, target, and whether a trade should be skipped.
5. A backtest or paper-trading layer proves whether the system makes money after costs.

Right now, the repo is at the first serious stage: training a CNN to classify future movement from synchronized multi-timeframe OHLCV windows.

## 2. How the current project actually works

The README describes a large quant system, but the code currently implements a narrower pipeline:

1. Raw OHLCV JSON is stored by timeframe in `data/raw/1min`, `data/raw/5min`, and `data/raw/1hr`.
2. `script/prepare_data.py` reads one stock's raw files, synchronizes the three timeframes around each 5-minute anchor candle, and creates:
   - `X_1min.npy`
   - `X_5min.npy`
   - `X_1hr.npy`
   - `y_labels.npy`
3. Labels are created from future 5-minute close movement:
   - `BUY` if 50-minute future return > `+0.5%`
   - `SELL` if 50-minute future return < `-0.5%`
   - `NEUTRAL` otherwise
4. `script/merge_data.py` merges processed arrays across multiple instruments into:
   - `data/master_training/MASTER_X1.npy`
   - `data/master_training/MASTER_X5.npy`
   - `data/master_training/MASTER_XH.npy`
   - `data/master_training/MASTER_y.npy`
5. `models/cnn/train.py` loads the master arrays and trains a 3-branch CNN.
6. `models/cnn/model.py` defines the CNN:
   - one branch for 1-minute data
   - one branch for 5-minute data
   - one branch for 1-hour data
   - branch embeddings are concatenated and classified into 3 classes

## 3. Exact current workflow

### Data creation

For each 5-minute anchor candle:

1. Take the last 60 candles from the 1-minute dataframe.
2. Take the last 60 candles from the 5-minute dataframe.
3. Take the last 60 candles from the 1-hour dataframe.
4. Normalize each window locally using min-max scaling per column.
5. Look 10 candles ahead on the 5-minute series.
6. Assign the label based on the future percentage move.

That produces three tensors of shape `(60, 6)` per sample, where the 6 columns are most likely OHLCV-style market fields from the JSON.

### Merge step

The processed arrays from each stock are appended into one large dataset, shuffled, and saved as the master training set.

Current master dataset:

- Samples: `3,17,097`
- Input shape per branch: `(60, 6)`
- Labels:
  - `0 = Neutral`: `264,411` (`83.38%`)
  - `1 = Buy`: `27,097` (`8.55%`)
  - `2 = Sell`: `25,589` (`8.07%`)

### Training step

`models/cnn/train.py` does the following:

1. Loads the master `.npy` arrays.
2. Adds a channel dimension so each branch becomes `(batch, 1, 60, 6)`.
3. Creates a `TensorDataset` and `DataLoader`.
4. Builds the `MultiTimeframeCNN`.
5. Uses weighted cross-entropy to compensate for heavy neutral-class dominance.
6. Trains for 30 epochs with:
   - Adam
   - weight decay
   - gradient clipping
   - learning-rate scheduler
7. Saves weights to `models/pattern_master_cnn.pth`.

## 4. How the CNN model works

`models/cnn/model.py` is a three-eye architecture.

Each eye:

1. Takes one timeframe input.
2. Applies two 2D convolution blocks.
3. Applies batch normalization and leaky ReLU.
4. Pools to a single feature vector of size `64`.

Then:

1. The three `64`-dimensional outputs are concatenated into `192` features.
2. A dense layer maps `192 -> 128`.
3. Dropout is applied.
4. The final dense layer maps `128 -> 3` for `Neutral / Buy / Sell`.

This is a valid baseline architecture for the current dataset.

## 5. What this means in practical terms

Today, the model is not learning "how to trade profitably."

It is learning a much narrower task:

"Given recent multi-timeframe OHLCV structure, predict whether price will move up, down, or sideways over roughly the next 50 minutes."

That is a useful first milestone, but it is not yet a complete trading system.

## 6. What you should focus on next

Focus order matters a lot here. The biggest mistake would be jumping into many extra models before the baseline is scientifically valid.

### Priority 1: Make the baseline trustworthy

You need:

1. Train/validation/test split
2. Time-based split, not random leakage across nearby windows
3. Metrics beyond training loss:
   - class-wise precision/recall/F1
   - confusion matrix
   - balanced accuracy
4. Saved experiment configs and results

If this is missing, you cannot tell whether the CNN actually works.

### Priority 2: Evaluate as a trading system, not just a classifier

You need a backtest layer that answers:

1. If the model predicts `Buy`, when do you enter?
2. What stop loss and take profit are used?
3. What is the holding period?
4. What are brokerage, slippage, and taxes?
5. What is max drawdown?
6. What is win rate, expectancy, Sharpe-like return quality, and profit factor?

Without this, a good classifier can still be a bad trading strategy.

### Priority 3: Fix the label design

The current labels are simple and reasonable for a baseline, but they are weak for a production trading system.

The main problems:

1. A fixed `0.5%` threshold may not match each stock's volatility.
2. It ignores transaction cost and slippage.
3. It does not encode risk-adjusted reward.
4. It does not distinguish easy trades from noisy moves.

Stronger future labels would be:

1. ATR-scaled movement labels
2. Triple-barrier labeling
3. Return buckets adjusted for cost
4. Meta-labeling for "take trade / skip trade"

### Priority 4: Build a proper research pipeline

You should have a clean sequence:

1. Data ingestion
2. Feature generation
3. Window generation
4. Labeling
5. Train/validation/test split
6. Model training
7. Offline evaluation
8. Backtest
9. Paper trading
10. Live trading only after the above is stable

### Priority 5: Add non-price context only after the baseline works

Your README mentions:

1. EMA, MACD, ADX, RSI, ATR, VWAP
2. candlestick patterns
3. sector and index context
4. options features
5. news or external factors

These are good additions, but they should come after the baseline CNN is measurable and reproducible.

Otherwise you will add complexity without learning what actually improved performance.

## 7. What I would not do yet

I would not build many separate models immediately.

Do not start with:

1. one CNN
2. one transformer
3. one sentiment model
4. one options model
5. one RL agent

all at once.

That usually creates noise, not progress.

Instead:

1. Make one strong CNN baseline.
2. Add one tabular feature model.
3. Compare them honestly.
4. Then consider ensembling if both add independent value.

## 8. Recommended target architecture

The clearest architecture for your vision is:

### Stage A: Pattern model

- Multi-timeframe CNN on normalized OHLCV windows

### Stage B: Quant feature model

- Gradient boosted trees or an MLP on engineered indicators, regime features, and market context

### Stage C: Meta decision model

- Takes outputs from Stage A and Stage B
- Predicts:
  - trade direction
  - confidence
  - whether to skip the trade

### Stage D: Risk and execution engine

- position sizing
- ATR stop
- take profit logic
- trade cooldown
- max daily loss
- no-trade filters during bad regimes

This is much closer to how the project could realistically become useful.

## 9. Exact future workflow I recommend

1. Collect raw multi-timeframe OHLCV for each instrument.
2. Clean timestamps, sort candles, remove duplicates, and verify continuity.
3. Build synchronized windows across 1-minute, 5-minute, and 1-hour data.
4. Create labels using volatility-aware logic and trading costs.
5. Split data by time into train, validation, and test periods.
6. Train the CNN on only the training period.
7. Evaluate on validation and test with class metrics and calibration.
8. Convert predictions into trade rules using confidence thresholds.
9. Run a proper backtest with costs, slippage, and position rules.
10. Compare against simple baselines:
   - buy and hold
   - moving average crossover
   - VWAP or breakout rules
11. Add engineered quant features and train a second model.
12. Combine both only if the second model adds out-of-sample value.
13. Run paper trading before any live execution.

## 10. Concrete issues in the current repo

These are the main gaps I would address next:

1. `models/cnn/train.py` has no validation or test loop.
2. The current workflow appears to shuffle all master samples together before model development, which is risky for time-series evaluation.
3. `script/prepare_data.py` is currently hardcoded to `reliance` raw file paths.
4. The training file imports `from model import MultiTimeframeCNN`, which depends on how the script is executed.
5. There is no backtesting or trade execution simulation yet.
6. There is no clear experiment tracking.
7. There is no explicit protection against train/test leakage at the market regime level.

## 11. Bottom line

The project idea is good, but the real near-term goal should be narrower:

Build a reliable research-grade baseline that can answer this question:

"Does synchronized multi-timeframe price structure contain enough signal to produce tradable predictions after costs?"

If the answer is yes, then the rest of the quant stack is worth building.

If the answer is no, that is still a valuable result, because it tells you to redesign labels, features, or the decision layer before scaling complexity.
