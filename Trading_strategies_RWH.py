import numpy as np
import pandas as pd
import yfinance as yf
import datetime as dt
from pylab import mpl, plt

plt.style.use("seaborn-v0_8-whitegrid")
mpl.rcParams['font.family'] = 'serif'


def load_raw_data(tickers, start_date, end_date):
    raw = yf.download(tickers, start_date, end_date)['Adj Close']
    raw = raw.reindex(columns=tickers)
    raw = pd.DataFrame(raw)
    raw.info()
    return raw


def random_walk_hypothesis(raw, symbol):
    data = pd.DataFrame(raw[symbol])
    lags = 5
    cols = []
    for lag in range(1, lags + 1):
        col = 'lag_{}'.format(lag)
        data[col] = data[symbol].shift(lag)
        cols.append(col)
    print(data.head(7))
    data.dropna(inplace=True)

    reg = np.linalg.lstsq(data[cols], data[symbol], rcond=-1)
    print(reg[0])
    plt.figure(figsize=(10, 6))
    plt.bar(cols, reg[0])
    plt.show()
    data['Perdiction'] = np.dot(data[cols], reg[0])
    data[[symbol, 'Perdiction']].iloc[-75:].plot(figsize=(10, 6))
    plt.show()
    return data, reg


if __name__ == '__main__':
    tickers = ['SPY', 'ALB']
    start_date = '2020-01-01'
    end_date = '2023-05-31'
    raw_ = load_raw_data(tickers, start_date, end_date)
    symbol = 'SPY'
    data_, reg_ = random_walk_hypothesis(raw_, symbol)