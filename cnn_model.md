# CNN Model Guide

## Goal

Yes. The correct strategy is:

Do not try to make the whole trading system perfect at once.

Make one thing strong first.

Right now that one thing is:

`multi-timeframe CNN that learns price structure from 1-minute, 5-minute, and 1-hour OHLCV windows`

If this part becomes reliable, then you can build the rest of the stack on top of it.

## What "best CNN" actually means

For this project, "best" does **not** mean:

1. biggest network
2. most layers
3. most indicators
4. most complicated architecture

It means:

1. clean data
2. good labels
3. no leakage
4. stable training
5. honest validation
6. useful predictions for trading

That is the standard you should hold.

## What you already have

You already have enough raw data in the repo to start properly:

1. `reliance`
2. `hdfc`
3. `tataMotors`
4. `nifty`

Timeframes already present:

1. `1min`
2. `5min`
3. `1hr`

So for now, do **not** waste time fetching more data.

Start by making the CNN excellent on the dataset you already have.

## Final objective of this phase

At the end of this phase, you should be able to answer:

`Can a CNN trained on synchronized multi-timeframe OHLCV windows predict future short-term directional movement better than chance, in a way that can later be turned into trades?`

That is the real mission.

## Step-by-step plan

## Step 1: Freeze the problem definition

Do not keep changing the task every day.

For the current CNN phase, lock the problem as:

1. Input:
   - last 60 candles of `1min`
   - last 60 candles of `5min`
   - last 60 candles of `1hr`
2. Features per candle:
   - `open`
   - `high`
   - `low`
   - `close`
   - `volume`
   - `oi`
3. Label:
   - `buy`
   - `sell`
   - `neutral`
4. Forecast horizon:
   - 10 future `5min` candles
   - roughly 50 minutes

Why this matters:

If you keep changing the target, you will never know whether the CNN improved or the task changed.

## Step 2: Build the dataset cleanly

The repo now supports processing all available raw stocks with one command.

Run:

```bash
./.venv/bin/python script/prepare_data.py --stocks all
```

What this does:

1. reads raw JSON from all three timeframes
2. aligns windows by 5-minute anchor timestamp
3. creates normalized windows
4. labels future movement
5. saves per-stock arrays to `data/processed/<stock>/`
6. saves `dataset_summary.json` per stock

After that, merge everything into the master dataset:

```bash
./.venv/bin/python script/merge_data.py
```

That creates:

1. `data/master_training/MASTER_X1.npy`
2. `data/master_training/MASTER_X5.npy`
3. `data/master_training/MASTER_XH.npy`
4. `data/master_training/MASTER_y.npy`

## Step 3: Understand what the CNN is learning

The CNN is not learning profit directly.

It is learning this narrower question:

`Given recent multi-timeframe structure, is price likely to move up, down, or sideways over the next 50 minutes?`

This matters because if you expect it to "become a full trader" immediately, you will evaluate it incorrectly.

## Step 4: Train the current baseline correctly

The training script is now runnable from the repo with explicit arguments.

Basic training command:

```bash
cd models/cnn
../../.venv/bin/python train.py --epochs 30 --batch-size 64 --lr 5e-4 --device cpu
```

If you have a usable GPU-enabled PyTorch environment later, use:

```bash
cd models/cnn
../../.venv/bin/python train.py --epochs 30 --batch-size 64 --lr 5e-4 --device cuda
```

What this training script currently does:

1. loads master arrays
2. adds the channel dimension
3. builds the 3-branch CNN
4. uses class-weighted cross-entropy
5. uses Adam with weight decay
6. uses gradient clipping
7. uses a learning-rate scheduler
8. saves `models/pattern_master_cnn.pth`

## Step 5: What to optimize first

Do these in order.

### 5.1 Data quality

This is the highest leverage area.

Focus on:

1. no missing candles inside windows
2. no duplicate timestamps
3. consistent timezone handling
4. clean OHLCV values
5. correct alignment between 1m, 5m, and 1h windows

If your data is wrong, a better CNN will only learn wrong patterns faster.

### 5.2 Labels

Your current labels are okay for a first baseline, but they are not final.

Current logic:

1. if future return > `+0.5%` -> `buy`
2. if future return < `-0.5%` -> `sell`
3. else -> `neutral`

This is acceptable now, but later you should test:

1. ATR-scaled thresholds
2. different horizons
3. cost-aware labels
4. triple-barrier labels

Do not change labels immediately.

First see whether the current setup produces any real signal.

### 5.3 Class imbalance

Your data is heavily neutral-dominated.

Approximate current distribution:

1. `neutral`: about `83%`
2. `buy`: about `8.5%`
3. `sell`: about `8%`

Because of this, a model can look good while mostly predicting neutral.

So you should care about:

1. buy recall
2. sell recall
3. precision on actionable classes
4. confusion matrix

Training loss alone is not enough.

### 5.4 Stability

To make the model strong, stability matters more than raw complexity.

You want:

1. similar behavior across runs
2. learning curves that make sense
3. no exploding gradients
4. no collapsing to only neutral predictions

## Step 6: What you should change next in code

This is the correct next implementation order.

### Phase A

Add proper train/validation/test splitting by time.

This is the single most important missing piece.

Why:

If nearby windows from the same periods leak across train and test, your reported performance will be misleading.

### Phase B

Add evaluation metrics:

1. confusion matrix
2. per-class precision
3. per-class recall
4. per-class F1
5. balanced accuracy

### Phase C

Save experiment artifacts:

1. model weights
2. config used
3. metrics
4. dataset version

### Phase D

Run ablation studies.

Examples:

1. only 5-minute branch
2. 1-minute + 5-minute
3. all three branches
4. with OI
5. without OI

This tells you what actually adds signal.

## Step 7: Hyperparameter tuning order

Do not tune everything at once.

Tune in this order:

1. label thresholds and horizon
2. learning rate
3. class weights
4. batch size
5. dropout
6. branch depth
7. hidden layer size

Why this order:

Bad labels cannot be fixed by architecture tuning.

## Step 8: Recommended experiment sequence

Run the CNN research like this:

### Experiment 1

Current architecture, current labels, current dataset.

Goal:

Check whether the system learns anything beyond the neutral majority.

### Experiment 2

Same architecture, better validation and metrics.

Goal:

See true out-of-sample behavior.

### Experiment 3

Different thresholds:

1. `0.3%`
2. `0.5%`
3. `0.7%`

Goal:

Find a better actionability/noise balance.

### Experiment 4

Different horizons:

1. 6 candles
2. 10 candles
3. 15 candles

Goal:

Find whether the CNN is stronger on shorter or slightly longer setups.

### Experiment 5

Architecture improvements only after the first four are measured.

Possible ideas:

1. more filters
2. residual blocks
3. branch-specific kernel shapes
4. attention after branch fusion

## Step 9: What the "best version" of this CNN phase looks like

You are done with the CNN phase when all of this is true:

1. dataset generation is repeatable
2. splits are time-safe
3. metrics are tracked
4. results are reproducible
5. CNN beats naive baselines
6. predictions can be translated into a testable trading rule

That is the finish line for this phase.

## Step 10: Exact commands to run now

### Rebuild processed stock datasets

```bash
./.venv/bin/python script/prepare_data.py --stocks all
```

### Merge into master training arrays

```bash
./.venv/bin/python script/merge_data.py
```

### Train the CNN baseline

```bash
cd models/cnn
../../.venv/bin/python train.py --epochs 30 --batch-size 64 --lr 5e-4 --device cpu
```

## What I changed for you

I updated:

1. `script/prepare_data.py`
2. `models/cnn/train.py`

### `script/prepare_data.py`

It now:

1. supports `--stocks all`
2. processes every available stock automatically
3. validates stock names
4. saves a `dataset_summary.json` per stock
5. is no longer hardcoded only to `reliance`

### `models/cnn/train.py`

It now:

1. uses repo-safe absolute paths
2. supports CLI args for epochs, batch size, learning rate, weight decay, and device
3. can be run directly from `models/cnn`

## My direct guidance to you

If you want to make this CNN as good as possible, your next mindset should be:

1. fix the data pipeline
2. lock the problem definition
3. measure honestly
4. improve one variable at a time
5. do not add five new models yet

That is how this becomes real instead of becoming a pile of ideas.

## What I recommend next

The next concrete coding step should be:

`add a proper train/validation/test split with evaluation metrics`

That is the most valuable next upgrade for the CNN phase.
