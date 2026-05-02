### The Core Vision
The vision is to build an institutional-grade Quantitative Trading Intelligence that moves beyond simple indicators (like RSI or Moving Averages). Instead, this project treats market data as a visual-temporal geometry problem.

Look in this project i am trying to build something that can be on par with the best trader in the world and bit of best algos in the world that make trades in tock market to make money not loose, yess i dont have the best resoucres but i am sure i can but an algo which can trade on behalf of me and make best decisions.

### How It Works
## 1. CNN Model
The project utilizes a Functional Multi-Input 2D-Convolutional Neural Network (CNN). Unlike standard models that look at one data stream, this project uses three specialized "Eyes" to process the market:

1. Micro Branch ->	1-Minute  ->	Detects immediate momentum, liquidity spikes, and entry precision.
2. Intraday Branch  ->	5-Minute  ->	Identifies session-level trends and local support/resistance zones.
3. Macro Branch  ->	1-Hour  ->	Provides the "Big Picture" context, ensuring trades align with the dominant trend.

## 2. Quant Model 
This would be capable to analyse and understand all of these below
1. Trend Features
• EMA 9, 21, 50, 200 — and their crossover states (golden cross, death cross)
• MACD line, signal line, histogram, divergence flag
• ADX (trend strength) — above 25 = strong trend, below 20 = choppy
• Supertrend indicator with ATR multiplier
• Price position relative to VWAP (above/below, % distance)
• Higher highs / higher lows detection (swing structure)
2. Momentum Features
• RSI (14) — with overbought/oversold flags and RSI divergence
• Stochastic RSI — faster momentum signal
• Rate of Change (ROC) over 5, 10, 20 periods
• Momentum oscillator — measures speed of price change
3. Volume Features
• Volume vs 20-day average ratio (volume spike detection)
• OBV (On-Balance Volume) — tracks smart money flow
• VWAP — volume weighted average price (intraday benchmark)
• Cumulative delta — buy volume minus sell volume pressure
• Delivery percentage (for positional trades)
4. Volatility Features
• ATR (Average True Range) — raw volatility in price terms
• Bollinger Bands — width, position of price within bands, squeeze
• India VIX value and its 5-day change
• Historical volatility vs implied volatility (options premium check)
5. Candlestick Pattern Features
• Doji, Hammer, Shooting Star, Engulfing patterns (encoded as 0/1 flags)
• Inside bar, outside bar, pin bar detection
• Gap up / gap down open detection and size
• Multiple candle patterns: Morning Star, Evening Star, Three White Soldiers
6. Options-Derived Features
• PCR (Put-Call Ratio) — sentiment indicator
• Max pain price — where most options expire worthless
• IV Rank and IV Percentile — is volatility cheap or expensive?
• Open Interest change at key strikes — where is money positioned?
7. Market Context Features
• Nifty 50 trend alignment — is the broad market supporting your trade?
• Sector strength — is the stock's sector outperforming or underperforming?
• Time-of-day encoding — markets behave differently at 9:30 AM vs 2:30 PM
• Day-of-week encoding — Mondays and Fridays have statistical tendencies
• Distance from key support/resistance levels (computed automatically)

In this quant we will also ahve a model or a thing like we would get data about the stock from online means news to see if any external factors are influencing the stock.

------------------------------------------------------------------------
Exact workflow is not really clear of what to do but it will be a cluster of ML models for sure to analyses all of those things and place trades in market which would actually make money.

Not Sure either how many models do i actually need for this but currently on the CNN model phase to train the CNN to be best at understanding patterns and all to get best results. 
What do you think??
