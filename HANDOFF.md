# Tech Stock Forecast — Handoff

## What this is
A Jupyter notebook that predicts next-day stock prices for 5 tech tickers (AAPL, MSFT, AMZN, GOOGL, TSLA) using 2019-2024 data. It's a **hybrid model** — classification (price up or down?) and regression (what will the price be?).

## Where things stand
The notebook is through **data preprocessing, feature engineering, and feature selection**. Two targets are ready. No model has been trained yet.

### Dataset
- Source: `./datasets/major-tech-stock-2019-2024.csv`
- ~6,140 rows after cleaning (5 tickers x ~1,228 trading days)
- Index is Date, not a column

### Features (7 selected)
| Feature | What it is |
|---------|-----------|
| MA_7 | 7-day moving average of Adj Close |
| Volume | Trading volume |
| Ticker | Encoded 0-4 via OrdinalEncoder |
| RSI_14 | Relative Strength Index (14-day) |
| Momentum_7d | 7-day % price change |
| Volatility_7d | 7-day annualized volatility |
| Daily_Return | Intraday % return (open to close) |

### Targets (in `trgt_y`)
- **Target** — next-day Adj Close (regression)
- **Direction** — 1 = up, 0 = down (classification, ~53/47 split, balanced)

### Decisions already made
- **Adj Close over Close** for all price-based features — corrects for stock splits (AAPL and TSLA both split in this period)
- **Dropped Open, High, Low, MA_30** — highly correlated with MA_7, kept MA_7 as the smoothed price signal
- **Kept Momentum_7d + RSI_14** despite 0.63 correlation — they measure different things (trend vs overbought/oversold)
- **NaN rows dropped** (not imputed) — only ~29 rows per ticker lost from rolling window warmup

## What needs to be done next
1. **Train/test split** — must be chronological, not random (this is time-series). Split per ticker (e.g. 80/20 by date)
2. **Feature scaling** — fit scaler on train only, transform both sets
3. **Train classifier** — Random Forest or XGBoost for Direction
4. **Train regressor** — Random Forest or XGBoost for Target
5. **Evaluate** — classification: accuracy/F1/confusion matrix; regression: MAE/RMSE/R²
6. **Combine outputs** — "Price goes up, predicted: $152.30"

## Watch out for
- **No random splitting.** Future data leaking into training will give fake-good results
- All rolling features use `groupby("Ticker").transform()` — this prevents cross-ticker data leakage
- The feature set lives in cell 5 as `useful_feats_X` — that's the one to use going forward
- `trgt_y` has both targets in one DataFrame — split by column name when feeding to each model
