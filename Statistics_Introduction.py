import math
import numpy as np
import scipy.stats as scs
import statsmodels.api as sm
from pylab import mpl, plt

plt.style.use('seaborn-v0_8-whitegrid')
mpl.rcParams['font.family'] = 'serif'


def Normality_test():
    print('The normal distribution can be considered the most important distribution in finance and one of the \n '
          'major statistical building blocks of financial theory. The following cornerstones of financial theory \n '
          'rest to a large extent on the assumption that returns of a financial instrument are normally distributed:')
    print('Portfolio theory: \n'
          'When stock returns are normally distributed, optimal portfolio choice can be cast into a setting where \n '
          'only the (expected) mean return and the variance of the returns (or the volatility) as well as the \n '
          'covariances between different stocks are relevant for an investment decision')
    print('Captial asset pricing model: \n'
          'When stock returns are normally distributed, prices of single stocls can be elegantly expressd in \n'
          'linear relationship to a broad market index; the relationship is generally expressed by a measure for \n'
          'the co-movement of a sigle stock with the market index called beta or β')
    print('Efficient market hypothesis: \n'
          'An efficient market is a market where prices reflect all available information, where "all" can be \n'
          'defined more narrowly or more widely (e.g., as in "all pubilcly available" information vs. including also \n'
          '"only privately available" information). If this hypothesis holds true, then stock prices fluctuate  \n'
          'randomly and returns are normally distributed')
    print('Option pricing theory: \n'
          'Brownian motion is the benchmark model for the  modeling of random price movements of financial \n'
          'instruments; the famous Black-Scholes-Merton option oricing formula uses a geometric Brownian motion as \n'
          'the model for a stocks random price fluctuations over time, leading to log-normally distributed \n'
          'prices and normally distributed returns')


def benchmark_case(S0, r, sigma, T, M, I):
    dt = T / M
    paths = np.zeros((M + 1, I))
    paths[0] = S0
    for t in range(1, M + 1):
        rand = np.random.standard_normal(I)
        rand = (rand - rand.mean()) / rand.std()
        paths[t] = paths[t - 1] * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * math.sqrt(dt) * rand)

    price = S0 * math.exp(r * T)
    print(price)
    print(paths[-1].mean())

    plt.figure(figsize=(10, 6))
    plt.plot(paths[:, :10])
    plt.xlabel('time steps')
    plt.ylabel('index level')
    plt.show()

    log_returns = np.log(paths[1:] / paths[:-1])
    print(log_returns.shape)
    print(log_returns.flatten().shape)

    return paths, log_returns


def print_statistics(array, M, sigma):
    sta = scs.describe(array.flatten())
    print('%14s %15s' % ('statistic', 'value'))
    print(30 * '-')
    print('%14s %15.f' % ('size', sta[0]))
    print('%14s %15.5f' % ('min', sta[1][0]))
    print('%14s %15.5f' % ('max', sta[1][1]))
    print('%14s %15.5f' % ('mean', sta[2]))
    print('%14s %15.5f' % ('std', np.sqrt(sta[3])))
    print('%14s %15.5f' % ('skew', sta[4]))
    print('%14s %15.5f' % ('kurtosis', sta[5]))

    mean = array.mean() * M + 0.5 * sigma ** 2
    std = array.std() * math.sqrt(M)
    print(mean, std)

    plt.figure(figsize=(10, 6))
    plt.hist(array.flatten(), bins=70, density=True, label='frequency', color='b')
    x = np.linspace(plt.axis()[0], plt.axis()[1])
    plt.plot(x, scs.norm.pdf(x, loc=r / M, scale=sigma / np.sqrt(M)), 'r', lw=2.0, label='pdf')
    plt.legend()
    plt.show()

    sm.qqplot(array.flatten()[::500], line='s')
    plt.xlabel('theoretical quantiles')
    plt.ylabel('sample quantiles')
    plt.show()

    return sta


def normality_tests(array):
    print('Skew of data set  %14.3f' % scs.skew(array))
    print('Skew test p-value  %14.3f' % scs.skewtest(array)[1])
    print('Kurt of data set  %14.3f' % scs.kurtosis(array))
    print('Kurt test p-value  %14.3f' % scs.kurtosistest(array)[1])
    print('Norm test p-value  %14.3f' % scs.normaltest(array)[1])


def hist(array):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))
    ax1.hist(array[-1], bins=30)
    ax1.set_xlabel('index level')
    ax1.set_ylabel('frequency')
    ax1.set_title('regular data')
    ax2.hist(np.log(array[-1]), bins=30)
    ax2.set_xlabel('log index level')
    ax2.set_title('log data')
    plt.show()


if __name__ == '__main__':
    Normality_test()
    S0 = 100
    r = 0.05
    sigma = 0.2
    T = 1.0
    M = 50
    I = 250000
    paths_, log_returns_ = benchmark_case(S0, r, sigma, T, M, I)
    sta_ = print_statistics(log_returns_, M, sigma)
    normality_tests(log_returns_.flatten())
    hist(paths_)
