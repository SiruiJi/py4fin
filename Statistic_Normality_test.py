import pandas as pd
import numpy as np
import yfinance as yf
import scipy.stats as scs
import statsmodels.api as sm
from pylab import mpl, plt

plt.style.use('seaborn-v0_8-whitegrid')
mpl.rcParams['font.family'] = 'serif'


def load_raw_data(tickers, start_date, end_date):
    raw = yf.download(tickers, start_date, end_date)['Adj Close']
    raw.reindex(columns=tickers)
    raw = pd.DataFrame(raw)
    return raw


def print_statistics(array):
    sta = scs.describe(array.flatten())
    print('%14s %15s' % ('statistic', 'value'))
    print(30 * '-')
    print('%14s %15.f' % ('size', sta[0]))
    print('%14s %15.5f' % ('min', sta[1][0]))
    print('%14s %15.5f' % ('max', sta[1][1]))
    print('%14s %15.5f' % ('mean', sta[2]))
    print('%14s %15.5f' % ('std', np.sqrt(sta[3])))
    print('%14s %15.5f' % ('skew', sta[4]))
    print('%14s %15.5f' % ('kurtosis', sta[5]))


def data_info(raw, tickers):
    (raw / raw.iloc[0] * 100).plot(figsize=(10, 6))
    plt.show()

    log_returns = np.log(raw / raw.shift(1))
    log_returns.hist(bins=50, figsize=(10, 8))

    for sym in tickers:
        print('\nResult for symbol {}'.format(sym))
        print(30 * '-')
        log_data = np.array(log_returns[sym].dropna())
        print_statistics(log_data)
        sm.qqplot(log_returns[sym].dropna(), line='s')
        plt.title(sym)
        plt.xlabel('theoretical quantiles')
        plt.ylabel('sample quantiles')
        plt.show()


if __name__ == '__main__':
    tickers = ['ABBV', 'GOOGL', 'JNJ', 'DLTR', 'HLT', 'JPM', 'DEO', 'PG', 'ALB', 'BA', 'NVDA', 'LUV', 'PEP', 'TSM',
               'SPY', '^VIX', 'GLD']
    start_date = '2015-01-01'
    end_date = '2023-05-08'
    raw_ = load_raw_data(tickers, start_date, end_date)
    data_info(raw_, tickers)