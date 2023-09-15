import pandas as pd
import numpy as np
import yfinance as yf
from pylab import plt, mpl
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

plt.style.use('seaborn-v0_8-whitegrid')
mpl.rcParams['font.family'] = 'serif'


def load_raw_data(tickers, start_date, end_date):
    raw = yf.download(tickers, start_date, end_date)['Adj Close']
    raw.reindex(columns=tickers)
    raw = pd.DataFrame(raw)
    return raw


def create_bins(raw, symbol, lags):
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

    mu = data['returns'].mean()
    v = data['returns'].std()
    bins = [mu - v, mu, mu + v]

    cols_bin = []
    for col in cols:
        col_bin = col + 'bin'
        data[col_bin] = np.digitize(data[col], bins=bins)
        cols_bin.append(col_bin)

    return data, cols_bin, bins


def randomized_train_test_split(data):
    train, test = train_test_split(data, test_size=0.5, shuffle=True)
    train = train.copy().sort_index()
    test = test.copy().sort_index()
    return train, test


def train_model(train, cols_bin):
    model = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=2*[250])
    model.fit(train[cols_bin], train['direction'])
    return model


def model_test(model, test, cols_bin):
    test['pos_dnn_sk'] = model.predict(test[cols_bin])
    test['strat_dnn_sk'] = test['pos_dnn_sk'] * test['returns']

    print(test[['returns', 'strat_dnn_sk']].sum().apply(np.exp))
    test[['returns', 'strat_dnn_sk']].cumsum().apply(np.exp).plot(figsize=(10, 6))
    plt.show()
    return test


def next_day_prediction(model, symbol, pred_start, pred_end, lags, bins):
    pred_data = pd.DataFrame(yf.download(symbol, pred_start, pred_end)['Adj Close'])
    pred_data['returns'] = np.log(pred_data / pred_data.shift(1))
    pred_data.dropna(inplace=True)
    pred_data['direction'] = np.sign(pred_data['returns']).astype(int)

    pcols = []
    for plag in range(1, lags + 1):
        col = 'lag_{}'.format(plag)
        pred_data[col] = pred_data['returns'].shift(plag - 1)
        pcols.append(col)
    pred_data.dropna(inplace=True)

    cols_bin = []
    for col in pcols:
        col_bin = col + 'bin'
        pred_data[col_bin] = np.digitize(pred_data[col], bins=bins)
        cols_bin.append(col_bin)

    pred_data['pred_pos'] = model.predict(pred_data[cols_bin])

    return pred_data


if __name__ == '__main__':
    tickers = ['SPY', 'AAPL']
    start_date = '2010-01-01'
    end_date = '2023-06-07'
    raw_ = load_raw_data(tickers, start_date, end_date)
    symbol = 'SPY'
    lags = 5
    data_, cols_bin_, bins = create_bins(raw_, symbol, lags)
    train_, test_ = randomized_train_test_split(data_)
    model_ = train_model(train_, cols_bin_)
    test_ = model_test(model_, test_, cols_bin_)
    pred_start = '2023-05-25'
    pred_end = '2023-06-07'
    pred_data_ = next_day_prediction(model_, symbol, pred_start, pred_end, lags, bins)
