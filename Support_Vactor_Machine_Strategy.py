from pylab import plt
import numpy as np
import yfinance as yf
import pandas as pd
from sklearn.svm import SVC


def load_data(ticker, Start_date, End_date, lags):
    prices = yf.download(ticker, start=Start_date, end=End_date)['Adj Close']
    df = pd.DataFrame({'Price': prices})
    df['Returns'] = np.log(df / df.shift())
    df.dropna(inplace=True)

    cols = []
    for lag in range(1, lags + 1):
        col = 'lag_{}'.format(lag)
        df[col] = np.sign(df['Returns'].shift(lag))
        cols.append(col)
    df.dropna(inplace=True)
    return df, cols, prices


def svm_train(df,cols,prices):
    model = SVC(gamma='auto')
    model.fit(df[cols], np.sign(df['Returns']))
    df['Prediction'] = model.predict(df[cols])
    df['Strategy'] = df['Prediction'] * df['Returns']
    compare = df[['Returns', 'Strategy']].cumsum().apply(np.exp)
    compare.plot()
    plt.show()
    cumreturn = np.log(prices / prices.shift(1)).cumsum()
    return model, cumreturn


def svm(ticker, pstart, End_date, lags, model):

    pprice = yf.download(ticker, start=pstart, end=End_date)['Adj Close']
    pdf = pd.DataFrame({'pPrice': pprice})
    pdf['pReturns'] = np.log(pdf / pdf.shift())
    pdf.dropna(inplace=True)

    pcols = []
    for plag in range(1, lags + 1):
        col = 'lag_{}'.format(plag)
        pdf[col] = np.sign(pdf['pReturns'].shift(plag - 1))
        pcols.append(col)
    pdf.dropna(inplace=True)

    pdf['Prediction'] = model.predict(pdf[pcols])
    return pdf


if __name__ == '__main__':
    ticker_ = 'JPM'
    Start_date_ = '2010-03-31'
    End_date_ = '2023-05-10'
    pstart = '2023-04-24'
    lags_ = 6
    df_, cols_, prices_ = load_data(ticker_, Start_date_, End_date_,lags_)
    model_, cumreturn_ = svm_train(df_, cols_, prices_)
    pdf_ = svm(ticker_, pstart, End_date_, lags_, model_)

