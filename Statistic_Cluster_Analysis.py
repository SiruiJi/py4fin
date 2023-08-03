import yfinance as yf
import numpy as np
import pandas as pd
from pylab import plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, AffinityPropagation
from sklearn import metrics
from sklearn import cluster
from statsmodels.tsa.stattools import coint
from sklearn.manifold import TSNE
import matplotlib.cm as cm
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import fcluster
from itertools import cycle


def load_raw_data(tickers, start_date, end_date):
    dataset = yf.download(tickers, start_date, end_date)['Adj Close']
    dataset.reindex(columns=tickers)
    dataset = pd.DataFrame(dataset)
    return dataset


def data_preparation(dataset):
    returns = pd.DataFrame(dataset.pct_change().mean() * 252)
    returns.columns = ['Returns']
    returns['Volatility'] = dataset.pct_change().std() * np.sqrt(252)
    data = returns
    scaler = StandardScaler().fit(data)
    rescaledDataset = pd.DataFrame(scaler.fit_transform(data), columns=data.columns, index=data.index)
    X = rescaledDataset
    return X


def find_optimal_number_of_clusters_for_kmean(X):
    distortions = []
    max_loop = 20
    for k in range(2, max_loop):
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(X)
        distortions.append(kmeans.inertia_)

    plt.figure(figsize=(15, 5))
    plt.plot(range(2, max_loop), distortions)
    plt.xticks([i for i in range(2, max_loop)], rotation=75)
    plt.grid(True)
    plt.show()

    silhouette_score = []
    for k in range(2, max_loop):
        kmeans = KMeans(n_clusters=k, random_state=10, n_init=10)
        kmeans.fit(X)
        silhouette_score.append(metrics.silhouette_score(X, kmeans.labels_, random_state=10))

    plt.figure(figsize=(15, 5))
    plt.plot(range(2, max_loop), silhouette_score)
    plt.xticks([i for i in range(2, max_loop)], rotation=75)
    plt.grid(True)
    plt.show()


def kmean_training(X, nclust):
    k_means = cluster.KMeans(n_clusters=nclust)
    k_means.fit(X)
    target_labels_ = k_means.predict(X)

    centroids = k_means.cluster_centers_
    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_subplot(111)
    scatter = ax.scatter(X.iloc[:, 0], X.iloc[:, 1], c=k_means.labels_, cmap="rainbow", label=X.index)
    ax.set_title('k_means result')
    ax.set_xlabel('Mean Return')
    ax.set_ylabel('Volatility')
    plt.plot(centroids[:, 0], centroids[:, 1], 'sg', markersize=11)
    plt.show()

    cluster_series = pd.Series(index=X.index, data=k_means.labels_.flatten())
    cluster_series_all = pd.Series(index=X.index, data=k_means.labels_.flatten())
    cluster_series = cluster_series[cluster_series != -1]
    plt.figure(figsize=(12, 7))
    plt.barh(
        range(len(cluster_series.value_counts())),
        cluster_series.value_counts()
    )
    plt.title('Cluster Member Counts')
    plt.xlabel('Stocks in Cluster')
    plt.ylabel('Cluster Number')
    plt.show()
    return k_means, centroids


def hierarchical_clustering_training(X, nclust):
    Z = linkage(X, method='ward')
    plt.figure(figsize=(10, 7))
    plt.title("Stocks Dendrograms")
    dendrogram(Z, labels=X.index)
    plt.show()
    distance_threshold = 13
    clusters = fcluster(Z, distance_threshold, criterion='distance')
    chosen_clusters = pd.DataFrame(data=clusters, columns=['cluster'])
    chosen_clusters['cluster'].unique()
    hc = AgglomerativeClustering(n_clusters=nclust, affinity='euclidean', linkage='ward')
    clust_labels1 = hc.fit_predict(X)
    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_subplot(111)
    scatter = ax.scatter(X.iloc[:, 0], X.iloc[:, 1], c=clust_labels1, cmap="rainbow")
    ax.set_title('Hierarchical Clustering')
    ax.set_xlabel('Mean Return')
    ax.set_ylabel('Volatility')
    plt.colorbar(scatter)
    return hc


def affinity_propagation(X):
    ap = AffinityPropagation()
    ap.fit(X)
    clust_labels2 = ap.predict(X)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    scatter = ax.scatter(X.iloc[:, 0], X.iloc[:, 1], c=clust_labels2, cmap="rainbow")
    ax.set_title('Affinity')
    ax.set_xlabel('Mean Return')
    ax.set_ylabel('Volatility')
    plt.colorbar(scatter)
    #plt.show()
    cluster_centers_indices = ap.cluster_centers_indices_
    labels = ap.labels_
    no_clusters = len(cluster_centers_indices)
    print('Estimated number of clusters: %d' % no_clusters)
    # Plot exemplars

    X_temp = np.asarray(X)
    plt.close('all')
    plt.figure(1)
    plt.clf()

    fig = plt.figure(figsize=(8, 6))
    colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
    for k, col in zip(range(no_clusters), colors):
        class_members = labels == k
        cluster_center = X_temp[cluster_centers_indices[k]]
        plt.plot(X_temp[class_members, 0], X_temp[class_members, 1], col + '.')
        plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col, markeredgecolor='k', markersize=14)
        for x in X_temp[class_members]:
            plt.plot([cluster_center[0], x[0]], [cluster_center[1], x[1]], col)

    plt.show()
    return ap


def model_evaluation(X, k_means, hc, ap):
    print("km", metrics.silhouette_score(X, k_means.labels_, metric='euclidean'))
    print("hc", metrics.silhouette_score(X, hc.fit_predict(X), metric='euclidean'))
    print("ap", metrics.silhouette_score(X, ap.labels_, metric='euclidean'))

def model_selection(model, X, dataset):
    cluster_series = pd.Series(index=X.index, data=model.fit_predict(X).flatten())
    cluster_series_all = pd.Series(index=X.index, data=model.fit_predict(X).flatten())
    cluster_series = cluster_series[cluster_series != -1]
    counts = cluster_series_all.value_counts()
    cluster_vis_list = list(counts[(counts < 25) & (counts > 1)].index)[::-1]

    CLUSTER_SIZE_LIMIT = 9999
    counts = cluster_series.value_counts()
    ticker_count_reduced = counts[(counts > 1) & (counts <= CLUSTER_SIZE_LIMIT)]
    print("Clusters formed: %d" % len(ticker_count_reduced))
    print("Pairs to evaluate: %d" % (ticker_count_reduced * (ticker_count_reduced - 1)).sum())
    for clust in cluster_vis_list[0:min(len(cluster_vis_list), 4)]:
        means = np.log(dataset.loc[:"2020-02-01", tickers].mean())
        data = np.log(dataset.loc[:"2020-02-01", tickers]).sub(means)
        data.plot(title='Stock Time Series for Cluster %d' % clust)
    plt.show()
    return data, ticker_count_reduced, cluster_series


def find_cointegrated_pairs(data, significance=0.05):
    n = data.shape[1]
    score_matrix = np.zeros((n, n))
    pvalue_matrix = np.zeros((n, n))
    keys = data.keys()
    pairs = []
    for i in range(1):
        for j in range(i + 1, n):
            S1 = data[keys[i]].dropna()
            S2 = data[keys[j]].dropna()
            common_index = S1.index.intersection(S2.index)
            S1 = S1.loc[common_index]
            S2 = S2.loc[common_index]

            result = coint(S1, S2)
            score = result[0]
            pvalue = result[1]
            score_matrix[i, j] = score
            pvalue_matrix[i, j] = pvalue
            if pvalue < significance:
                pairs.append((keys[i], keys[j]))
    return score_matrix, pvalue_matrix, pairs


def outcome_visualization(ticker_count_reduced, dataset, X, cluster_series, tickers, centroids):
    cluster_dict = {}
    for i, which_clust in enumerate(ticker_count_reduced.index):
        score_matrix, pvalue_matrix, pairs = find_cointegrated_pairs(
            dataset[tickers]
        )
        cluster_dict[which_clust] = {}
        cluster_dict[which_clust]['score_matrix'] = score_matrix
        cluster_dict[which_clust]['pvalue_matrix'] = pvalue_matrix
        cluster_dict[which_clust]['pairs'] = pairs
    pairs = []
    for clust in cluster_dict.keys():
        pairs.extend(cluster_dict[clust]['pairs'])
    print("Number of pairs found : %d" % len(pairs))
    print("In those pairs, there are %d unique tickers." % len(np.unique(pairs)))

    stocks = np.unique(pairs)
    X_df = pd.DataFrame(index=X.index, data=X).T
    in_pairs_series = cluster_series.loc[stocks]
    stocks = list(np.unique(pairs))
    X_pairs = X_df.T.loc[stocks]
    X_tsne = TSNE(learning_rate=50, perplexity=1, random_state=1337).fit_transform(X_pairs)
    plt.figure(1, facecolor='white', figsize=(16, 8))
    plt.clf()
    plt.axis('off')
    for pair in pairs:
        # print(pair[0])
        ticker1 = pair[0]
        loc1 = X_pairs.index.get_loc(pair[0])
        x1, y1 = X_tsne[loc1, :]
        # print(ticker1, loc1)

        ticker2 = pair[0]
        loc2 = X_pairs.index.get_loc(pair[1])
        x2, y2 = X_tsne[loc2, :]

        plt.plot([x1, x2], [y1, y2], 'k-', alpha=0.3, c='gray');

    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], s=220, alpha=0.9, c=in_pairs_series.values, cmap=cm.Paired)
    plt.title('T-SNE Visualization of Validated Pairs');

    # zip joins x and y coordinates in pairs
    for x, y, name in zip(X_tsne[:, 0], X_tsne[:, 1], X_pairs.index):
        label = name

        plt.annotate(label,  # this is the text
                     (x, y),  # this is the point to label
                     textcoords="offset points",  # how to position the text
                     xytext=(0, 10),  # distance from text to points (x,y)
                     ha='center')  # horizontal alignment can be left, right or center

    plt.plot(centroids[:, 0], centroids[:, 1], 'sg', markersize=11)
    plt.show()
    return pairs


if __name__ == '__main__':
    tickers = ['AAPL', 'MSFT', 'AMZN', 'NVDA', 'GOOGL', 'GOOG', 'META', 'TSLA', 'UNH', 'XOM', 'JNJ', 'JPM', 'V',
           'PG', 'LLY', 'MA', 'HD', 'MRK', 'AVGO', 'CVX', 'PEP', 'ABBV', 'KO', 'PFE', 'COST', 'MCD', 'CRM', 'WMT',
           'TMO', 'CSCO', 'BAC', 'ABT', 'ACN', 'LIN', 'AMD', 'CMCSA', 'ADBE', 'DIS', 'NFLX', 'WFC', 'TXN', 'ORCL', 'VZ',
           'DHR', 'NEE', 'PM', 'BMY', 'RTX', 'NKE', 'HON', 'COP', 'LOW', 'UPS', 'INTC', 'AMGN', 'UNP', 'INTU', 'SPGI',
           'MDT', 'QCOM', 'SBUX', 'T', 'IBM', 'BA', 'PLD', 'GE', 'CAT', 'ELV', 'GS', 'MS', 'ISRG', 'MDLZ', 'LMT',
           'AMAT', 'NOW', 'BKNG', 'GILD', 'BLK', 'DE', 'SYK', 'AXP', 'TJX', 'ADP', 'CVS', 'ADI', 'C', 'VRTX', 'MMC',
           'AMT', 'TMUS', 'CB', 'MO', 'SCHW', 'REGN', 'ZTS', 'SO', 'PGR', 'LRCX', 'CI', 'BSX', 'MU', 'PYPL',
           'BDX', 'DUK', 'ETN', 'EOG', 'TGT', 'SLB', 'CSX', 'CME', 'AON', 'CL', 'NOC', 'HUM', 'EQIX', 'ITW', 'WM',
           'SNPS', 'APD', 'ICE', 'ORLY', 'CMG', 'KLAC', 'HCA', 'CDNS', 'ATVI', 'MCK', 'SHW', 'MMM', 'FDX', 'EW', 'PXD',
           'MPC', 'GIS', 'MCO', 'PNC', 'NSC', 'CCI', 'FCX', 'MSI', 'ROP', 'DG', 'GD', 'KMB', 'SRE', 'AZO', 'EMR', 'MAR',
           'GM', 'DXCM', 'PSX', 'PSA', 'F', 'VLO', 'EL', 'MNST', 'AEP', 'AJG', 'BIIB', 'MRNA', 'APH', 'OXY', 'FTNT',
           'NXPI', 'USB', 'D', 'ADSK', 'JCI', 'ECL', 'PH', 'TRV', 'AIG', 'TDG', 'MCHP', 'TFC', 'ADM', 'CTAS', 'EXC',
           'CTVA', 'HSY', 'IDXX', 'TT', 'COF', 'TEL', 'STZ', 'O', 'CPRT', 'HES', 'YUM', 'IQV', 'PCAR', 'MSCI', 'HLT',
           'AFL', 'SYY', 'DOW', 'A', 'CNC', 'XEL', 'WMB', 'WELL', 'ANET', 'ROST', 'CHTR', 'LHX', 'PAYX', 'ON',
            'VRSK', 'NUE', 'SPG', 'ILMN', 'DHI', 'NEM', 'AME', 'ED', 'MET', 'EA', 'KMI', 'DVN', 'DLTR', 'RMD',
           'KR', 'CSGP', 'FIS', 'CTSH', 'PPG', 'AMP', 'ROK', 'FAST', 'VICI', 'KHC', 'DD', 'PEG', 'ALL', 'CMI', 'GWW',
           'BK', 'PRU', 'MTD', 'RSG', 'BKR', 'HAL', 'KEYS', 'WEC', 'ABC', 'AWK', 'LEN', 'ODFL', 'KDP',
           'ZBH', 'DFS', 'ACGL', 'PCG', 'OKE', 'GPN', 'ANSS', 'IT', 'HPQ', 'WBD', 'VMC', 'EFX', 'EIX', 'WST', 'ES',
           'ALB', 'DLR', 'FANG', 'ULTA', 'MLM', 'APTV', 'TSCO', 'SBAC', 'AVB', 'GLW', 'PWR', 'WTW', 'CBRE', 'EBAY',
           'STT', 'TROW', 'URI', 'IR', 'CHD', 'CDW', 'LYB', 'FTV', 'DAL', 'GPC', 'ENPH', 'HIG', 'WBA', 'MKC', 'CAH',
           'TTWO', 'BAX', 'AEE', 'WY', 'DTE', 'VRSN', 'MTB', 'IFF', 'PODD', 'FE', 'ALGN', 'EQR', 'ETR', 'CTRA', 'STE',
           'HOLX', 'CLX', 'FSLR', 'EXR', 'DRI', 'PPL', 'FICO', 'LH', 'DOV', 'INVH', 'COO', 'MPWR', 'OMC', 'TDY', 'LVS',
           'HPE', 'XYL', 'NDAQ', 'CNP', 'ARE', 'EXPD', 'BR', 'K', 'FITB', 'RJF', 'LUV', 'VTR', 'WAB', 'FLT', 'CMS',
           'RCL', 'NVR', 'BALL', 'MAA', 'ATO', 'CAG', 'RF', 'MOH', 'SEDG', 'TYL', 'TRGP', 'CINF', 'HWM', 'SJM', 'GRMN',
           'SWKS', 'LW', 'PFG', 'IRM', 'STLD', 'WAT', 'MRO', 'UAL', 'IEX', 'AMCR', 'NTRS', 'PHM', 'BRO', 'RVTY', 'HBAN',
           'TSN', 'TER', 'FDS', 'J', 'DGX', 'IPG', 'EPAM', 'NTAP', 'CBOE', 'RE', 'JBHT', 'PTC', 'EXPE', 'AKAM', 'LKQ',
           'PAYC', 'BBY', 'BG', 'SNA', 'AES', 'ZBRA', 'AVY', 'EVRG', 'ESS', 'EQT', 'CFG', 'FMC', 'SYF', 'TXT', 'LNT',
           'AXON', 'POOL', 'CF', 'TECH', 'MGM', 'UDR', 'MOS', 'PKG', 'INCY', 'STX', 'WDC', 'HST', 'CHRW', 'SWK', 'LYV',
           'TRMB', 'WRB', 'NDSN', 'CPT', 'TAP', 'KMX', 'MAS', 'HRL', 'APA', 'IP', 'L', 'KIM', 'ETSY', 'NI',
           'BWA', 'VTRS', 'DPZ', 'LDOS', 'TFX', 'PEAK', 'CCL', 'CE', 'JKHY', 'WYNN', 'CPB', 'MKTX', 'HSIC', 'CRL',
           'TPR', 'EMN', 'GEN', 'QRVO', 'GL', 'KEY', 'MTCH', 'FOXA', 'ALLE', 'CDAY', 'JNPR', 'PNR', 'ROL', 'AAL', 'CZR',
           'REG', 'BBWI', 'PNW', 'AOS', 'FFIV', 'UHS', 'XRAY', 'NRG', 'HII', 'BIO', 'HAS', 'RHI', 'PARA', 'GNRC', 'WHR',
           'NWSA', 'WRK', 'AAP', 'BEN', 'BXP', 'CTLT', 'IVZ', 'AIZ', 'FRT', 'VFC', 'SEE', 'NCLH', 'DXC', 'ALK', 'DVA',
           'CMA', 'MHK', 'RL', 'ZION', 'FOX', 'NWL', 'LNC', 'NWS', 'DISH']

    start_date = '2020-01-01'
    end_date = '2023-08-03'
    dataset_ = load_raw_data(tickers, start_date, end_date)
    X_ = data_preparation(dataset_)
    find_optimal_number_of_clusters_for_kmean(X_)
    nclust = 4
    k_means_, centroids_ = kmean_training(X_, nclust)
    hc_ = hierarchical_clustering_training(X_, nclust)
    ap_ = affinity_propagation(X_)
    model_evaluation(X_, k_means_, hc_, ap_)
    model = ap_
    data_, ticker_count_reduced_, cluster_series_ = model_selection(model, X_, dataset_)
    socre_matrix_, pvalue_matrix_, pairs_ = find_cointegrated_pairs(data_, significance=0.05)
    paris1_ = outcome_visualization(ticker_count_reduced_, dataset_, X_, cluster_series_, tickers, centroids_)