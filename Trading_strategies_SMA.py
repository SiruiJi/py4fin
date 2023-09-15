import yfinance as yf
import numpy as np
import pandas as pd
from pylab import mpl, plt

plt.style.use("seaborn-v0_8-whitegrid")
mpl.rcParams['font.family'] = 'serif'


def load_raw_data(ticker, start_date, end_date):
    raw = yf.download(ticker, start_date, end_date)['Adj Close']
    raw = raw.reindex(columns=ticker)
    raw.info()
    raw = (pd.DataFrame(raw).dropna())
    return raw


def simple_moving_average(raw, symbol, SMA1, SMA2):
    raw['SMA1'] = raw[symbol].rolling(SMA1).mean()
    raw['SMA2'] = raw[symbol].rolling(SMA2).mean()
    raw.plot(figsize=(10, 6))
    plt.show()
    raw['Position'] = np.where(raw['SMA1'] > raw['SMA2'], 1, -1)
    raw.tail()
    ax = raw.plot(secondary_y='Position', figsize=(10, 6))
    ax.get_legend().set_bbox_to_anchor((0.25, 0.85))
    plt.show()


def Vectorized_backtesting(raw, symbol):
    raw['Returns'] = np.log(raw[symbol] / raw[symbol].shift(1))
    raw['Strategy'] = raw['Position'].shift(1) * raw['Returns']
    raw.round(4).head()
    raw.dropna(inplace=True)
    ret = np.exp(raw[['Returns', 'Strategy']].sum())
    std = raw[['Returns', 'Strategy']].std() * 252 ** 0.5
    print(ret)
    print(std)
    ax = raw[['Returns', 'Strategy']].cumsum().apply(np.exp).plot(figsize=(10, 6))
    raw['Position'].plot(ax=ax, secondary_y='Position', style='--')
    ax.get_legend().set_bbox_to_anchor((0.25, 0.85))
    plt.show()


from itertools import product


def Optimization(raw, symbol):
    sma1 = range(20, 61, 4)
    sma2 = range(180, 281, 10)
    results = pd.DataFrame()
    for oSMA1, oSMA2 in product(sma1, sma2):
        data = pd.DataFrame(raw[symbol])
        data.dropna(inplace=True)
        data['Returns'] = np.log(data[symbol] / data[symbol].shift(1))
        data['oSMA1'] = data[symbol].rolling(oSMA1).mean()
        data['oSMA2'] = data[symbol].rolling(oSMA2).mean()
        data.dropna(inplace=True)
        data['Position'] = np.where(data['oSMA1'] > data['oSMA2'], 1, -1)
        data['Strategy'] = data['Position'].shift(1) * data['Returns']
        data.dropna(inplace=True)
        perf = np.exp(data[['Returns', 'Strategy']].sum())
        results = results.append(pd.DataFrame(
                    {'oSMA1': oSMA1, 'oSMA2': oSMA2,
                     'MARKET': perf['Returns'],
                     'STRATEGY': perf['Strategy'],
                     'OUT': perf['Strategy'] - perf['Returns']},
                    index=[0]), ignore_index=True)
    results.info()
    print(results.sort_values('OUT', ascending=False).head(7))
    return results, data


if __name__ == '__main__':
    ticker = ['SPY', 'ALB']
    start_date = '2013-01-01'
    end_date = '2023-05-31'
    raw_ = load_raw_data(ticker, start_date, end_date)
    symbol = 'ALB'
    SMA1 = 52
    SMA2 = 180
    simple_moving_average(raw_, symbol, SMA1, SMA2)
    Vectorized_backtesting(raw_, symbol)
    results_, data_ = Optimization(raw_, symbol)  # Applying varying Simple Moving Average (SMA) periods will alter
    # the start date of open positions, leading to variations in both the strategy return and the benchmark market
    # return.
