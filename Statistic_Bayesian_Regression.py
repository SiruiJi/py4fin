import pandas as pd
import numpy as np
import yfinance as yf
from pylab import mpl, plt
import pymc3 as pm
from pymc3.distributions.timeseries import GaussianRandomWalk

plt.style.use('seaborn-v0_8-whitegrid')
mpl.rcParams['font.family'] = 'serif'


def load_raw_data(tickers, start_date, end_date):
    raw = yf.download(tickers, start_date, end_date)['Adj Close']
    raw.reindex(columns=tickers)
    raw = pd.DataFrame(raw)
    return raw


def data_info(raw, symbols):
    data = raw[symbols]
    data = data / data.iloc[0]
    data.info()
    print(data.iloc[-1] / data.iloc[0] - 1)
    print(data.corr())
    data.plot(figsize=(10, 6))
    plt.show()

    mpl_dates = mpl.dates.date2num(data.index.to_pydatetime())
    plt.figure(figsize=(10, 6))
    plt.scatter(data['SPY'], data['GOOGL'], c=mpl_dates, marker='o', cmap='coolwarm')
    plt.xlabel('SPY')
    plt.ylabel('GOOGL')
    plt.colorbar(ticks=mpl.dates.DayLocator(interval=250),
                 format=mpl.dates.DateFormatter('%d %b %y'))
    plt.show()

    return data, mpl_dates


def bayesian_reg(data, mpl_dates):
    with pm.Model() as model:
        alpha = pm.Normal('alpha', mu=0, sd=20)
        beta = pm.Normal('beta', mu=0, sd=20)
        sigma = pm.Uniform('sigma', lower=0, upper=50)
        y_est = alpha + beta * data['GOOGL'].values
        likelihood = pm.Normal('SPY', mu=y_est, sd=sigma, observed=data['SPY'].values)
        start = pm.find_MAP()
        step = pm.NUTS()
        "Even though variable likelihood and step are not used elsewhere in the code, it's crucial for the underlying "
        "computation done by PyMC3. The pm.sample() function implicitly uses those function to estimate the parameters."
        trace = pm.sample(cores=1)

    print(pm.summary(trace))
    pm.traceplot(trace)
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.scatter(data['SPY'], data['GOOGL'], c=mpl_dates, marker='o', cmap='coolwarm')
    plt.xlabel('SPY')
    plt.ylabel('GOOGL')
    for i in range(len(trace)):
        plt.plot(data['SPY'], trace['alpha'][i] + trace['beta'][i] * data['GOOGL'])
    plt.colorbar(ticks=mpl.dates.DayLocator(interval=250),
                 format=mpl.dates.DateFormatter('%d %b %y'))
    plt.show()

    return model


if __name__ == '__main__':
    tickers = ['SPY', 'GOOGL', 'BA', 'DLTR', 'HLT', 'JPM']
    start_date = '2023-01-01'
    end_date = '2023-05-08'
    raw_ = load_raw_data(tickers, start_date, end_date)
    symbols = ['SPY', 'GOOGL']
    data_, mpl_dates = data_info(raw_, symbols)
    model_ = bayesian_reg(data_, mpl_dates)
