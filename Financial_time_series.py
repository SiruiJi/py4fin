import yfinance as yf
import numpy as np
import pandas as pd
from pylab import mpl, plt

plt.style.use("seaborn-v0_8-whitegrid")
mpl.rcParams['font.family'] = 'serif'


def load_data(tickers, start_date, end_date):
    data = yf.download(tickers, start_date, end_date)['Adj Close']
    data = data.reindex(columns=tickers)
    data.info()
    return data


def data_info(data):
    data.head()
    data.tail()
    data.describe().round(2)
    data.aggregate([min,
                    np.mean,
                    np.std,
                    np.median,
                    max]
                   ).round(2)
    data.plot(figsize=(10, 12), subplots=True)
    plt.show()
    instruments = ['AbbVie Inc.', 'Albemarle Cororation', 'The Boeing Company', 'Diageo plc', 'Dollar Tree Inc.',
                   'SPDR Gold Shares', 'Alphabet Inc.', 'Hilton Worldwide Holdings Inc.', 'Johnson & Johnson',
                   'JPMorgan Chase & Co.', 'Southwest Airlines Co.', 'NVIDIA Corporation', 'PepsiCo, Inc.',
                   'The Procter & Gamble', 'SPDR S&P 500 ETF Trust', 'Taiwan Semiconductor Manufacturing Company',
                   'CBOE Volatility Index']
    for ric, name in zip(data, instruments):
        print('{:8s} | {}'.format(ric, name))


def rets_Changes_over_time(data):
    data.diff().head()
    data.diff().mean()
    data.pct_change().round(3).head()
    data.pct_change().mean().plot(kind='bar', figsize=(10, 6))
    plt.show()
    rets = np.log(data / data.shift(1))
    rets.head().round(3)
    rets.cumsum().apply(np.exp).plot(figsize=(10, 6))
    plt.show()
    return rets


def resampling(data, rets):
    data.resample('W', label='right').last().head()  # weekly
    data.resample('M', label='right').last().head()
    rets.cumsum().apply(np.exp).resample('M', label='right').last().plot(figsize=(10, 6))
    plt.show()


def rolling_Statistic(data):
    sym = 'NVDA'
    rdata = pd.DataFrame(data[sym]).dropna()
    rdata.tail()
    window = 20
    rdata['min'] = data[sym].rolling(window=window).min()
    rdata['mean'] = data[sym].rolling(window=window).mean()
    rdata['std'] = data[sym].rolling(window=window).std()
    rdata['median'] = data[sym].rolling(window=window).median()
    rdata['max'] = data[sym].rolling(window=window).max()
    rdata['ewma'] = data[sym].ewm(halflife=0.5, min_periods=window).mean()
    rdata.dropna().head()
    ax = rdata[['min', 'mean', 'max']].iloc[-200:].plot(
        figsize=(10, 6), style=['g--', 'r--', 'g--'], lw=0.8)
    data[sym].iloc[-200:].plot(ax=ax, lw=2.0)
    plt.show()
    return rdata


def simple_moving_avarages(data):
    sym = 'NVDA'
    sdata = pd.DataFrame(data[sym]).dropna()
    sdata['SMA1'] = data[sym].rolling(window=42).mean()
    sdata['SMA2'] = data[sym].rolling(window=252).mean()
    sdata[[sym, 'SMA1', 'SMA2']].tail()
    sdata[[sym, 'SMA1', 'SMA2']].plot(figsize=(10, 6))
    sdata.dropna(inplace=True)
    sdata['positions'] = np.where(sdata['SMA1'] > sdata['SMA2'],
                                  1,
                                  -1)
    ax = sdata[[sym, 'SMA1', 'SMA2', 'positions']].plot(figsize=(10, 6),
                                                        secondary_y='positions')
    ax.get_legend().set_bbox_to_anchor((0.25, 0.85))
    plt.show()
    return sdata


def corr_analysis(data):
    sym = ['SPY', '^VIX']
    rhodata = pd.DataFrame(data[sym]).dropna()
    rhodata.tail()
    rhodata.plot(subplots=True, figsize=(10, 6))
    plt.show()
    rhodata.loc[:'2015-05-01'].plot(secondary_y='^VIX', figsize=(10, 6))
    plt.show()
    return rhodata


def logarithmic_returns(data):
    sym = ['SPY', '^VIX']
    logrets = np.log(data[sym] / data[sym].shift(1))
    logrets.head()
    logrets.dropna(inplace=True)
    logrets.plot(subplots=True, figsize=(10, 6))
    plt.show()
    pd.plotting.scatter_matrix(logrets,
                               alpha=0.2,
                               diagonal='hist',
                               hist_kwds={'bins': 35},
                               figsize=(10, 6))
    plt.show()


def ols_regression(rets):
    rets = rets.dropna()
    reg = np.polyfit(rets['SPY'], rets['^VIX'], deg=1)
    ax = rets.plot(kind='scatter', x='SPY', y='^VIX', figsize=(10, 6))
    ax.plot(rets['SPY'], np.polyval(reg, rets['SPY']), 'r', lw=2)
    plt.show()
    beta = reg[0]
    alpha = reg[1]
    print(beta, alpha)


def correlation(rets):
    rets.corr()
    ax = rets['SPY'].rolling(window=252).corr(rets['^VIX']).plot(figsize=(10, 6))
    ax.axhline(rets.corr().iloc[0, 1], c='r')
    plt.show()


def high_frequency_data(tick):
    tick['Mid'] = tick.mean(axis=1)
    tick['Mid'].plot(figsize=(10, 6))
    plt.show()
    tick_resam = tick.resample(rule='5min', label='right').last()
    tick_resam.head()
    tick_resam['Mid'].plot(figsize=(10, 6))
    plt.show()


if __name__ == '__main__':
    tickers = ['ABBV', 'GOOGL', 'JNJ', 'DLTR', 'HLT', 'JPM', 'DEO', 'PG', 'ALB', 'BA', 'NVDA', 'LUV', 'PEP', 'TSM',
               'SPY', '^VIX', 'GLD']
    start_date = '2015-01-01'
    end_date = '2023-05-08'
    data_ = load_data(tickers, start_date, end_date)
    #data_info(data_)
    #rets_ = rets_Changes_over_time(data_)
    #resampling(data_, rets_)
    #rdata_ = rolling_Statistic(data_)
    #sdata_ = simple_moving_avarages(data_)
    #rhodata_ = corr_analysis(data_)
    #logarithmic_returns(data_)
    #ols_regression(rets_)
    #correlation(rets_)
    tick = pd.read_csv('C:/Users/Sirui/Desktop/fxcm_eur_usd_tick_data.csv', index_col=0, parse_dates=True)
    high_frequency_data(tick)





