import numpy as np
import pandas as pd
import yfinance as yf
from pylab import mpl, plt
from sklearn import linear_model
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

plt.style.use("seaborn-v0_8-whitegrid")
mpl.rcParams['font.family'] = 'serif'


def load_raw_data(tickers, start_date, end_date):
    raw = yf.download(tickers, start_date, end_date)['Adj Close']
    raw = raw.reindex(columns=tickers)
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

    cols_bin = []
    for col in cols:
        col_bin = col + 'bin'
        data[col_bin] = np.digitize(data[col], bins=[0])
        cols_bin.append(col_bin)

    return data, cols_bin


def classification_models(data, cols_bin):
    models = {
        'log_reg': linear_model.LogisticRegression(C=1),
        'guass_nb': GaussianNB(),
        'svm': SVC(C=1)
    }

    mfit = {model: models[model].fit(data[cols_bin],
                                     data['direction'])
            for model in models.keys()}

    for model in models.keys():
        data['pos_' + model] = models[model].predict(data[cols_bin])

    sel = []
    for model in models.keys():
        col = 'strat_' + model
        data[col] = data['pos_' + model] * data['returns']
        sel.append(col)
    sel.insert(0, 'returns')

    print(data[sel].sum().apply(np.exp))
    data[sel].cumsum().apply(np.exp).plot(figsize=(10, 6))
    plt.show()

    return models, data


def next_day_predict(models, symbol, pred_start, pred_end, lags):
    pred_data = pd.DataFrame(yf.download(symbol, pred_start, pred_end)['Adj Close'])
    pred_data['returns'] = np.log(pred_data / pred_data.shift(1))
    pred_data.dropna(inplace=True)

    pcols = []
    for plag in range(1, lags + 1):
        col = 'lag_{}'.format(plag)
        pred_data[col] = pred_data['returns'].shift(plag - 1)
        pcols.append(col)
    pred_data.dropna(inplace=True)

    cols_bin = []
    for col in pcols:
        col_bin = col + 'bin'
        pred_data[col_bin] = np.digitize(pred_data[col], bins=[0])
        cols_bin.append(col_bin)

    for model in models.keys():
        pred_data['pred_' + model] = models[model].predict(pred_data[cols_bin])

    return pred_data


if __name__ == '__main__':
    tickers = ['SPY', 'BA']
    start_date = '2020-01-01'
    end_date = '2023-06-04'
    raw_ = load_raw_data(tickers, start_date, end_date)
    symbol = 'SPY'
    lags = 5
    data_, cols_bin_ = create_bins(raw_, symbol, lags)
    models_, data_ = classification_models(data_, cols_bin_)
    pred_start = '2023-05-10'
    pred_end = '2023-06-04'
    pred_data_ = next_day_predict(models_, symbol, pred_start, pred_end, lags)
