import numpy as np
import pandas as pd
import yfinance as yf
from pylab import mpl, plt
from sklearn import linear_model
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

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


def sequential_train_test_split(data):
    split = int(len(data) * 0.5)
    train = data.iloc[:split].copy()
    test = data.iloc[split:].copy()
    return train, test


def randomized_train_test_split(data):
    train, test = train_test_split(data, test_size=0.5, shuffle=True)
    train = train.copy().sort_index()
    test = test.copy().sort_index()
    return train, test


def train_models(train, cols_bin):
    models = {
        'log_reg': linear_model.LogisticRegression(C=1),
        'guass_nb': GaussianNB(),
        'svm': SVC(C=1)
    }

    mfit = {model: models[model].fit(train[cols_bin],
                                     train['direction'])
            for model in models.keys()}

    return models


def model_test(models, test, cols_bin):

    for model in models.keys():
        test['pos_' + model] = models[model].predict(test[cols_bin])

    sel = []
    for model in models.keys():
        col = 'strat_' + model
        test[col] = test['pos_' + model] * test['returns']
        sel.append(col)
    sel.insert(0, 'returns')

    print(test[sel].sum().apply(np.exp))
    test[sel].cumsum().apply(np.exp).plot(figsize=(10, 6))
    plt.show()

    return test


if __name__ == '__main__':
    tickers = ['SPY', 'NVDA']
    start_date = '2020-01-01'
    end_date = '2023-06-06'
    raw_ = load_raw_data(tickers, start_date, end_date)
    symbol = 'SPY'
    lags = 5
    data_, cols_bin_, bins = create_bins(raw_, symbol, lags)
    # train_, test_ = sequential_train_test_split(data_)
    train_, test_ = randomized_train_test_split(data_)
    models_ = train_models(train_, cols_bin_)
    test_ = model_test(models_, test_, cols_bin_)