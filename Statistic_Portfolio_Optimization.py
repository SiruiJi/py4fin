import pandas as pd
import numpy as np
import yfinance as yf
from pylab import mpl, plt
import scipy.optimize as sco
import scipy.interpolate as sci

plt.style.use('seaborn-v0_8-whitegrid')
mpl.rcParams['font.family'] = 'serif'


def load_raw_data(tickers, start_date, end_date):
    raw = yf.download(tickers, start_date, end_date)['Adj Close']
    raw.reindex(columns=tickers)
    raw = pd.DataFrame(raw)
    return raw


def data_info(raw, symbols):
    data = raw[symbols]
    rets = np.log(data / data.shift(1))
    rets.hist(bins=40, figsize=(10, 6))
    print(rets.mean() * 252)
    print(rets.cov() * 252)
    return rets


def port_ret(weights):
    return np.sum(rets.mean() * weights) * 252


def port_vol(weights):
    return np.sqrt(np.dot(weights.T, np.dot(rets.cov() * 252, weights)))


def min_func_sharpe(weights):
    """The negative value of the Sharpe ratio is minimized to derive at the maximum value and the optimal portfolio"""
    return -port_ret(weights) / port_vol(weights)


def maximum_sharpe_ratio(noa):
    eweights = np.array(noa * [1. / noa, ])  # set the equal weights at the beginning
    opts = sco.minimize(min_func_sharpe,
                        eweights, method='SLSQP',
                        bounds=tuple((0, 1) for x in range(noa)),
                        constraints=({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
                        )
    print(opts['x'].round(3))
    print(port_ret(opts['x']))
    print(port_vol(opts['x']))
    SR = port_ret(opts['x']) / port_vol(opts['x'])
    print(SR)

    return opts


def minimum_volatility(noa):
    eweights = np.array(noa * [1. / noa, ])
    optv = sco.minimize(port_vol, eweights, method='SLSQP',
                        bounds=tuple((0, 1) for x in range(noa)),
                        constraints=({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
                        )
    print(optv['x'].round(3))
    print(port_ret(optv['x'].round(3)))
    print(port_vol(optv['x'].round(3)))
    SR = port_ret(optv['x']) / port_vol(optv['x'])
    print(SR)
    return optv


def efficient_frontier(noa, opts, optv):
    prets = []
    pvols = []
    for p in range(10000):
        weights = np.random.random(noa)
        weights /= np.sum(weights)
        prets.append(port_ret(weights))
        pvols.append(port_vol(weights))
    prets = np.array(prets)
    pvols = np.array(pvols)

    plt.figure(figsize=(10, 6))
    plt.scatter(pvols, prets, c=prets / pvols, marker='o', cmap='coolwarm')
    plt.xlabel('expected volatility')
    plt.ylabel('expected return')
    plt.colorbar(label='Sharpe ratio')
    plt.show()

    trets = np.linspace(0.05, 0.2, 50)
    tvols = []

    eweights = np.array(noa * [1. / noa, ])
    for tret in trets:
        res = sco.minimize(port_vol, eweights, method='SLSQP',
                           bounds=tuple((0, 1) for x in range(noa)),
                           constraints=({'type': 'eq', 'fun': lambda x: port_ret(x) - tret},
                                        dict(type='eq', fun=lambda x: np.sum(x) - 1)))
        tvols.append(res['fun'])
    tvols = np.array(tvols)

    plt.figure(figsize=(10, 6))
    plt.scatter(pvols, prets, c=prets / pvols, marker='.', alpha=0.8, cmap='coolwarm')
    plt.plot(tvols, trets, 'b', lw=4.0)
    plt.plot(port_vol(opts['x']), port_ret(opts['x']), 'y*', markersize=15.0)
    plt.plot(port_vol(optv['x']), port_ret(optv['x']), 'r*', markersize=15.0)
    plt.xlabel('expected volatility')
    plt.ylabel('expected return')
    plt.colorbar(label='Sharpe ratio')
    plt.show()

    ind = np.argmin(tvols)
    evols = tvols[ind:]
    erets = trets[ind:]
    tck = sci.splrep(evols, erets)

    def f(x):
        """Efficient frontier function (splines approximation)."""
        return sci.splev(x, tck, der=0)

    def df(x):
        """First derivative of efficient frontier function"""
        return sci.splev(x, tck, der=1)

    def equations(p, rf=0.037):
        eq1 = rf - p[0]
        eq2 = rf + p[1] * p[2] - f(p[2])
        eq3 = p[1] - df(p[2])
        return eq1, eq2, eq3

    opt = sco.fsolve(equations, [0.01, 0.5, 0.15])
    plt.figure(figsize=(10, 6))
    plt.scatter(pvols, prets, c=(prets - 0.0037) / pvols, marker='.', cmap='coolwarm')
    plt.plot(evols, erets, 'b', lw=4.0)
    cx = np.linspace(0.0, 0.3)
    plt.plot(cx, opt[0] + opt[1] * cx, 'r', lw=1.5)
    plt.plot(opt[2], f(opt[2]), 'y*', markersize=15.0)
    plt.grid(True)
    plt.axhline(0, color='k', ls='--', lw=2.0)
    plt.axvline(0, color='k', ls='--', lw=2.0)
    plt.xlabel('expected volatility')
    plt.ylabel('expected return')
    plt.colorbar(label='Sharpe ratio')
    plt.show()

    res_ = sco.minimize(port_vol, eweights, method='SLSQP',
                        bounds=tuple((0, 1) for x in range(noa)),
                        constraints=({'type': 'eq', 'fun': lambda x: port_ret(x) - f(opt[2])},
                                     dict(type='eq', fun=lambda x: np.sum(x) - 1))
                        )
    print(res_['x'].round(3))
    print(port_ret(res_['x']))
    print(port_vol(res_['x']))
    SR = port_ret(res_['x']) / port_vol(res_['x'])
    print(SR)


if __name__ == '__main__':
    tickers = ['ABBV', 'GOOGL', 'JNJ', 'DLTR', 'HLT', 'JPM', 'DEO', 'PG', 'ALB', 'BA', 'NVDA', 'LUV', 'PEP', 'TSM',
               'SPY', '^VIX', 'GLD']
    start_date = '2015-01-01'
    end_date = '2023-05-08'
    raw_ = load_raw_data(tickers, start_date, end_date)
    symbols = ['ABBV', 'GOOGL', 'DLTR', 'JPM', 'DEO', 'PG', 'ALB', 'BA', 'NVDA', 'PEP', 'SPY', '^VIX', 'GLD']
    rets = data_info(raw_, symbols)
    noa = len(symbols)
    opts_ = maximum_sharpe_ratio(noa)
    optv_ = minimum_volatility(noa)
    efficient_frontier(noa, opts_, optv_)
