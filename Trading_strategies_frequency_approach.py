import numpy as np
import pandas as pd
import yfinance as yf
from pylab import mpl, plt

plt.style.use("seaborn-v0_8-whitegrid")
mpl.rcParams['font.family'] = 'serif'


def load_raw_data(tickers, start_date, end_date):
    raw = yf.download(tickers, start_date, end_date)['Adj Close']
    raw = raw.reindex(columns=tickers)
    raw = pd.DataFrame(raw)
    return raw


def highlight_max(s):
    is_max = s == s.max()
    return ['background_color: yellow' if v else '' for v in is_max]


def data_info(raw, symbol, lags):
    data = pd.DataFrame(raw[symbol])
    data['returns'] = np.log(data / data.shift(1))
    data.dropna(inplace=True)
    data['direction'] = np.sign(data['returns']).astype(int)
    cols = []
    for lag in range(1, lags + 1):
        col = 'lag_{}'.format(lag)
        data[col] = data['returns'].shift(lag)
        cols.append(col)
    data.dropna(inplace=True)

    cols_bin = []
    for col in cols:
        col_bin = col + '_bin'
        data[col_bin] = np.digitize(data[col], bins=[0])
        cols_bin.append(col_bin)

    print(data[cols_bin + ['direction']].head())
    grouped = data.groupby(cols_bin + ['direction'])
    print(grouped.size())

    res = grouped['direction'].size().unstack(fill_value=0)
    print(res.style.apply(highlight_max, axis=1))
    print(res)

    data['pos_freq'] = np.where(data[cols_bin].sum(axis=1) == 2, -1, 1)
    print((data['direction'] == data['pos_freq']).value_counts())
    data['strat_freq'] = data['pos_freq'] * data['returns']
    print(data[['returns', 'strat_freq']].sum().apply(np.exp))
    data[['returns', 'strat_freq']].cumsum().apply(np.exp).plot(figsize=(10, 6))
    plt.show()

    return data, cols


if __name__ == '__main__':
    tickers = ['SPY', 'NVDA']
    start_date = '2018-01-01'
    end_date = '2023-06-02'
    raw_ = load_raw_data(tickers, start_date, end_date)
    symbol = 'SPY'
    lags = 2
    data_, cols_ = data_info(raw_, symbol, lags)
