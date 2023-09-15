import numpy as np
import pandas as pd
import yfinance as yf
from pylab import mpl, plt
from sklearn.cluster import KMeans

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
    data.dropna(inplace=True)

    return data, cols


def cluster(data, cols):
    model = KMeans(n_clusters=2, random_state=0)
    model.fit(data[cols])
    data['clus_result'] = model.predict(data[cols])
    data['pos_clus'] = np.where(data['clus_result'] == 1, -1, 1)

    print(data['pos_clus'])
    plt.figure(figsize=(10, 6))
    plt.scatter(data[cols].iloc[:, 0], data[cols].iloc[:, 1], c=data['pos_clus'], cmap='coolwarm')
    plt.show()
    data['strat_clus'] = data['pos_clus'] * data['returns']
    print(data[['returns', 'strat_clus']].sum().apply(np.exp))
    print((data['direction'] == data['pos_clus']).value_counts())
    data[['returns', 'strat_clus']].cumsum().apply(np.exp).plot(figsize=(10, 6))
    plt.show()

    return model


def next_day_predict(symbol, pred_start, pred_end, lags, model):
    pred_data = pd.DataFrame(yf.download(symbol, pred_start, pred_end)['Adj Close'])
    pred_data['returns'] = np.log(pred_data / pred_data.shift(1))
    pred_data.dropna(inplace=True)

    pcols = []
    for plag in range(1, lags + 1):
        col = 'lag_{}'.format(plag)
        pred_data[col] = pred_data['returns'].shift(plag - 1)
        pcols.append(col)
    pred_data.dropna(inplace=True)

    pred_data['prediction'] = model.predict(pred_data[pcols])
    return pred_data


if __name__ == '__main__':

    tickers = ['ABBV', 'GOOGL', 'JNJ', 'DLTR', 'HLT', 'JPM', 'DEO', 'PG', 'ALB', 'BA', 'NVDA', 'LUV', 'PEP', 'TSM',
               'SPY', '^VIX', 'GLD']
    start_date = '2018-01-01'
    end_date = '2023-06-02'
    raw_ = load_raw_data(tickers, start_date, end_date)
    symbol = 'SPY'
    lags = 2
    data_, cols_ = data_info(raw_, symbol, lags)
    model_ = cluster(data_, cols_)
    pred_start = '2023-05-20'
    pred_end = '2023-06-02'
    pred_data_ = next_day_predict(symbol, pred_start, pred_end, lags, model_)
