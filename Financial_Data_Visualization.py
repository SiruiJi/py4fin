import matplotlib as mpl
# a = mpl.__version__
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import yfinance as yf
import numpy as np
import pandas as pd
import cufflinks as cf
import plotly.offline as plyo
plt.style.use("seaborn-v0_8")
mpl.rcParams['font.family'] = 'serif'


def load_data(ticker, start_date, end_date):
    dataset = yf.download(ticker, start=start_date, end=end_date)
    return dataset


def data_info(dataset):
    df = pd.DataFrame(dataset)
    df.info()
    quotes = df[['Open', 'High', 'Low', 'Adj Close']]
    quotes.tail()
    return df, quotes


def financial_plot(quotes):
    qf = cf.QuantFig(
        quotes,
        title=ticker,
        legend='top',
        name='EUR/USD'
    )

    plyo.iplot(
        qf.iplot(asFigure=True),
        image='png',
        filename='qf_01'
    )

    qf.add_bollinger_bands(periods=15,
                           boll_std=2)
    plyo.iplot(
        qf.iplot(asFigure=True),
        image='png',
        filename='qf_02'
    )

    qf.add_rsi(periods=14,
               showbands=False)
    plyo.iplot(
        qf.iplot(asFigure=True),
        image='png',
        filename='qf_03'
    )

    plt.show()


if __name__ == '__main__':
    ticker = 'NVDA'
    start_date = '2023-01-26'
    end_date = '2023-05-26'
    dataset_ = load_data(ticker, start_date, end_date)
    df_, quotes_ = data_info(dataset_)
    financial_plot(quotes_)
