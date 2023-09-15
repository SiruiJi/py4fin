import numpy as np
import pandas as pd
import yfinance as yf
from pylab import mpl, plt
from sklearn.linear_model import LinearRegression

plt.style.use("seaborn-v0_8-whitegrid")
mpl.rcParams['font.family'] = 'serif'


def load_raw_data(tickers, start_date, end_date):
    raw = yf.download(tickers, start_date, end_date)['Adj Close']
    raw = raw.reindex(columns=tickers)
    raw = pd.DataFrame(raw)
    raw.info()
    return raw


def data_info(raw, symbol, lags):
    data = pd.DataFrame(raw[symbol])
    data['returns'] = np.log(data / data.shift(1))
    data.dropna(inplace=True)
    data['direction'] = np.sign(data['returns']).astype(int)
    print(data.head())
    data['returns'].hist(bins=35, figsize=(10, 6))
    plt.show()

    cols = []
    for lag in range(1, lags + 1):
        col = 'lag_{}'.format(lag)
        data[col] = data['returns'].shift(lag)
        cols.append(col)
    print(data.head())

    data.dropna(inplace=True)
    data.plot.scatter(x='lag_1', y='lag_2', c='returns', cmap='coolwarm', figsize=(10, 6), colorbar=True)
    plt.axvline(0, c='r', ls='--')
    plt.axhline(0, c='r', ls='--')
    plt.show()
    return data, cols


def linear_regression(data, cols):
    model1 = LinearRegression()
    model2 = LinearRegression()
    data['pos_ols_1'] = model1.fit(data[cols], data['returns']).predict(data[cols])
    data['pos_ols_2'] = model2.fit(data[cols], data['direction']).predict(data[cols])
    print(data[['pos_ols_1', 'pos_ols_2']].head())
    data[['pos_ols_1', 'pos_ols_2']] = np.where(data[['pos_ols_1', 'pos_ols_2']] > 0, 1, -1)
    print(data['pos_ols_1'].value_counts())
    print(data['pos_ols_2'].value_counts())
    print((data['pos_ols_1'].diff() != 0).sum())
    print((data['pos_ols_2'].diff() != 0).sum())

    data['strat_ols_1'] = data['pos_ols_1'] * data['returns']
    data['strat_ols_2'] = data['pos_ols_2'] * data['returns']
    print(data[['returns', 'strat_ols_1', 'strat_ols_2']].sum().apply(np.exp))
    print((data['direction'] == data['pos_ols_1']).value_counts())
    print((data['direction'] == data['pos_ols_2']).value_counts())
    data[['returns', 'strat_ols_1', 'strat_ols_2']].cumsum().apply(np.exp).plot(figsize=(10, 6))
    plt.show()
    return model1, model2


def next_day_predict(symbol, pred_start, pred_end, lags, model1, model2):
    pred_data = pd.DataFrame(yf.download(symbol, pred_start, pred_end)['Adj Close'])
    pred_data['pred_returns'] = np.log(pred_data / pred_data.shift(1))
    pred_data.dropna(inplace=True)

    pcols = []
    for plag in range(1, lags+1):
        col = 'lag_{}'.format(plag)
        pred_data[col] = pred_data['pred_returns'].shift(plag - 1)
        pcols.append(col)
    pred_data.dropna(inplace=True)

    pred_data['prediction1'] = model1.predict(pred_data[pcols])
    pred_data['prediction2'] = model2.predict(pred_data[pcols])

    return pred_data


if __name__ == '__main__':
    tickers = ['ABBV', 'GOOGL', 'JNJ', 'DLTR', 'HLT', 'JPM', 'DEO', 'PG', 'ALB', 'BA', 'NVDA', 'LUV', 'PEP', 'TSM',
               'SPY', '^VIX', 'GLD', 'DIS']
    start_date = '2018-01-01'
    end_date = '2023-06-02'
    raw_ = load_raw_data(tickers, start_date, end_date)
    symbol = 'DIS'
    lags = 2
    data_, cols_ = data_info(raw_, symbol, lags)
    model1_, model2_ = linear_regression(data_, cols_)
    pred_start = '2023-07-26'
    pred_end = '2023-08-02'
    pred_data_ = next_day_predict(symbol, pred_start, pred_end, lags, model1_, model2_)


