import numpy as np
import pandas as pd
from yahooquery import Ticker
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import backtesting as bts
import matplotlib.pyplot as plt

effective_shares = 50
LOOKBACK = 210
SYMBOL = "GOOG" #^GSPC IS SPX ^DJI IS DOW JONES NASDAQ IS ^IXIC
START_DATE = (pd.Timestamp.today() - pd.Timedelta(days=700)).strftime("%Y-%m-%d")

tickers = Ticker(SYMBOL)
hist = tickers.history(start=START_DATE, interval="1h").reset_index()
data = hist[hist["symbol"] == SYMBOL][["date", "close", ]]
data.set_index("date", inplace=True)
prices = data["close"]
trades = []
kappa_min = 0.02
kappa = []
half_lives = []
kappa_dates = []
max_holding_date = 30

in_trade = False
entry_index = None
entry_direction = 00

Z_ENTRY_LONG = -1.5
Z_ENTRY_SHORT = 1.5
Z_EXIT = 0.4
trades = []

Q = np.array([[1e-6, 0],
              [0, 1e-6]])


x_state = np.array([prices.iloc[LOOKBACK], 0.0])
P = np.eye(2)
kalman_level = []
kalman_slope = []
kalman_dates = []
R = 5e-2

snr_values = []
snr_dates = []
for t in range(LOOKBACK,len(prices)):

    window = prices.iloc[t-LOOKBACK:t]
    mean = window.mean()
    std = window.std()

    x = window.values - mean
    x_lag = x[:-1]
    x_next = x[1:]

    phi = np.polyfit(x_lag, x_next, 1)[0]
    k = -np.log(abs(phi)) if 0 < abs(phi) < 1 else 0
    mean_reversion_okay = k > kappa_min
    half_life = np.log(2)/k if k > 0  else np.inf
    kappa.append(k)
    half_lives.append(half_life)
    kappa_dates.append(prices.index[t])


    price_today = prices.iloc[t]
    z = (price_today - mean) / std if std > 0 else 0
    Returns = window.pct_change().dropna()

    mu = Returns.mean()
    sd = Returns.std()
    snr = abs(mu) / sd if sd > 0 else 0


    F = np.array([[1, 1],
                  [0, 1]])

    x_pred = F @ x_state
    P_pred = F @ P @ F.T + Q

    H = np.array([[1, 0]])
    y = price_today - H @ x_pred
    S = H @ P_pred @ H.T + R
    K = P_pred @ H.T / S

    x_state = x_pred + (K.flatten() * y)
    P = (np.eye(2) - K @ H) @ P_pred

    kalman_level.append(x_state[0])
    kalman_slope.append(x_state[1])

    kalman_dates.append(prices.index[t])
    snr_values.append(snr)
    snr_dates.append(prices.index[t])

    sigma = sd
    kalman_snr = abs(x_state[1])/sigma if sigma > 0 else 0
    TREND_SNR = 0.11
    KALMAN_SNR = 0.3
    is_trending = (snr > TREND_SNR) and (kalman_snr > KALMAN_SNR)
    trend_dir = np.sign(x_state[1])


    if mean_reversion_okay and not in_trade:
        entry_half_life = half_life
        entry_kappa = k

        if z <= -Z_ENTRY_LONG and not is_trending:
            in_trade = True
            entry_index = t
            entry_direction = 1
            entry_price = price_today
            entry_z = z
            entry_date = prices.index[t]
            max_hold = int(min(half_life, max_holding_date))

        elif z >= Z_ENTRY_SHORT and not is_trending:
            in_trade = True
            entry_index = t
            entry_direction = -1
            entry_price = price_today
            entry_z = z
            entry_date = prices.index[t]
            max_hold = int(min(half_life, max_holding_date))



    if in_trade:
        time_in_trade = t - entry_index

        if abs(z) <= Z_EXIT or time_in_trade >= max_hold:
            exit_reason = "z_exit" if abs(z) <= Z_EXIT else "max_hold"
            exit_price = prices.iloc[t]
            exit_date = prices.index[t]
            pnl = effective_shares * entry_direction * (exit_price - entry_price)

            trades.append({
            "entry_date": entry_date,
            "exit_date": exit_date,
            "direction": entry_direction,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "pnl": pnl,
            "holding_time": time_in_trade,
            "entry_z": entry_z,
            "exit_z": z,
            "exit_reason": exit_reason,
            "half_life_at_entry": entry_half_life,
            "kappa_at_entry": k,
            "abs_entry_z": abs(entry_z),
            })

            in_trade = False
            entry_index = None
            entry_direction = 0



trade_df = pd.DataFrame(trades)
trade_df.groupby("exit_reason")["pnl"].agg(["count", "mean", "std", "min", "max"])
trade_df["hl_bucket"] = pd.qcut(trade_df["half_life_at_entry"], 4, labels=["fast", "medium", "slow", "very_slow"])
trade_df.groupby("hl_bucket")["pnl"].agg(["count", "mean", "std"])

trade_df["z_bucket"] = pd.qcut(trade_df["abs_entry_z"], 4,  labels=["small", "medium", "large", "extreme"])
trade_df.groupby("z_bucket")["pnl"].mean()

heat_map = trade_df.pivot_table(
    values="pnl",
    index="hl_bucket",
    columns="z_bucket",
    aggfunc="mean"
)

print(heat_map)

equity = pd.Series(0.0, index=prices.index)

for _, tr in trade_df.iterrows():
    equity.loc[tr["exit_date"]:] += tr["pnl"]

cum_equity = equity
cum_max = cum_equity.cummax()
drawdown = cum_equity - cum_max

START_CAPITAL = 100000
bh_shares = START_CAPITAL / prices.iloc[0]
bh_equity = bh_shares * prices


equity = pd.Series(
    START_CAPITAL,
    index=prices.index,
    dtype=float
)

for _, tr in trade_df.iterrows():
    equity.loc[tr["exit_date"]:] += tr["pnl"]

cum_max = equity.cummax()
drawdown = equity - cum_max


fig, axes = plt.subplots(
    4, 1,
    figsize=(14, 10),
    sharex=True,
    gridspec_kw={"height_ratios": [1, 0.6, 2, 0.6]}
)


axes[0].plot(equity.index, equity, color="blue", label="Equity")

axes[0].fill_between(
    equity.index,
    equity,
    cum_max,
    where=(equity < cum_max),
    color="red",
    alpha=0.25,
    label="Drawdown"
)

axes[0].set_ylabel("Equity ($)")
axes[0].ticklabel_format(style="plain", axis="y", useOffset=False)
axes[0].legend()
axes[0].grid(alpha=0.3)
axes[0].set_title("OU Strategy Backtest Report")


wins = trade_df[trade_df["pnl"] > 0]
losses = trade_df[trade_df["pnl"] <= 0]

axes[1].scatter(
    wins["exit_date"],
    wins["pnl"],
    color="green",
    marker="^",
    label="Wins"
)

axes[1].scatter(
    losses["exit_date"],
    losses["pnl"],
    color="red",
    marker="v",
    label="Losses"
)

axes[1].axhline(0, color="black", linestyle="--", linewidth=0.5)
axes[1].set_ylabel("Trade PnL ($)")
axes[1].legend()
axes[1].grid(alpha=0.3)


axes[2].plot(prices.index, prices, color="black", linewidth=1)

longs = trade_df[trade_df["direction"] == 1]
shorts = trade_df[trade_df["direction"] == -1]

axes[2].scatter(
    longs["entry_date"],
    longs["entry_price"],
    color="green",
    marker="^",
    label="Long Entry"
)

long_exits = trade_df[trade_df["direction"] == 1]
axes[2].scatter(
    long_exits["exit_date"],
    long_exits["exit_price"],
    color="green",
    marker="x",
    label="Long Exit"
)

axes[2].scatter(
    shorts["entry_date"],
    shorts["entry_price"],
    color="red",
    marker="v",
    label="Short Entry"
)

short_exits = trade_df[trade_df["direction"] == -1]
axes[2].scatter(
    short_exits["exit_date"],
    short_exits["exit_price"],
    color="red",
    marker="x",
    label="Short Exit"
)

axes[2].set_ylabel("Price")
axes[2].legend()
axes[2].grid(alpha=0.3)


volume = hist[hist["symbol"] == SYMBOL].set_index("date")["volume"]

axes[3].bar(
    volume.index,
    volume,
    color="gray",
    alpha=0.5
)


bh_equity = prices
bh_pnl = prices.iloc[-1] - prices.iloc[0]


axes[3].set_ylabel("Volume")
axes[3].grid(alpha=0.3)

plt.tight_layout()
plt.show()

buy_dates = []
buy_prices = []
sell_prices = []
sell_dates = []


for tr in trades:
    if tr["direction"] == 1:
       buy_dates.append(tr["entry_date"])
       buy_prices.append(tr["entry_price"])

kappa_series = pd.Series(kappa, index=kappa_dates)
half_live_series = pd.Series(half_lives, index=kappa_dates)


total_trades = len(trade_df)

wins = trade_df[trade_df["pnl"] > 0]
losses = trade_df[trade_df["pnl"] <= 0]

win_rate = len(wins) / total_trades if total_trades > 0 else np.nan
avg_win = wins["pnl"].mean()
avg_loss = losses["pnl"].mean()

expectancy = (
    win_rate * avg_win + (1 - win_rate) * avg_loss
)


returns = trade_df["pnl"]

sharpe = (
    returns.mean() / returns.std()
) * np.sqrt(252) if returns.std() > 0 else np.nan

max_dd = (equity - equity.cummax()).min()


print("===== STRATEGY STATS =====")
print(f"Total Trades     : {total_trades}")
print(f"Win Rate         : {win_rate:.2%}")
print(f"Avg Win ($)      : {avg_win:.2f}")
print(f"Avg Loss ($)     : {avg_loss:.2f}")
print(f"Expectancy ($)   : {expectancy:.2f}")
print(f"Sharpe (trade)   : {sharpe:.2f}")
print(f"Max Drawdown ($) : {max_dd:.2f}")

print("Long entries:", np.sum(trade_df["direction"] == 1))
print("Short entries:", np.sum(trade_df["direction"] == -1))
print("\n" + "-" * 40)
print(f"FINAL Buy & Hold PnL: {bh_pnl * effective_shares:.3f}")
print(f"FINAL OU EQUITY {equity.iloc[-1] - START_CAPITAL}")
print(f"Trend:,{trend_dir},Slope: {x_state[1]}, SNR: {snr:.4f}")
