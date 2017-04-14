import csv

import quandl

from SlidingWindow import *

quandl.ApiConfig.api_key = "DqLVWRStVw_hyQnnQvyW"

# dim is the length of each window, default 10


def analyze_SP500(dim=10, num_days=100):
	# use for analyzing entire sp500
	sp500 = open('SP500.csv', 'rt')
	pd_map = {}
	i = 0
	for row in csv.reader(sp500, delimiter=','):
		PDs = get_PD_stock(row[0], num_days=num_days)
		if PDs is not None:
			# make_plot(row[0], PDs)
			pd_map[row[0]] = PDs[1]
			i += 1
		if i == 5:
			calc_bottleneck_dists(pd_map)

def get_PD_stock(ticker, dim=10, num_days=100):
	# gather the data
	try:
		data = quandl.get("GOOG/NASDAQ_" + ticker, returns="numpy", rows=100)
	except quandl.errors.quandl_error.NotFoundError:
		print(ticker)
		return None
	x = [y[1] for y in data]
    	
	# get the sliding window vectors
	X = getSlidingWindowNoInterp(x, dim)

	# do TDA and PCA
	PDs = doRipsFiltration(X, 1)
	return PDs

# pd_map is a map of ticker to persistence diagram
def calc_bottleneck_dists(pd_map):
	keys = list(pd_map.keys())
	with open('bottleneck_dists.csv', 'wb') as csvfile:
		for i in range(0, len(keys)):
			ticker1 = keys[i]
			for j in range(i, len(keys)):
				ticker2 = keys[j]
				row = [ticker1, ticker2, getBottleneckDist(pd_map[ticker1], pd_map[ticker2])]
				csvfile.writerow(row)



def make_plot(ticker, PDs):
	pca = PCA(n_components = 2)
	Y = pca.fit_transform(X)
	eigs = pca.explained_variance_
	# Plot original signal, PCA of the embedding, and persistence diagram
	filename = "./diagrams/" + ticker + '.png'
	c = plt.get_cmap('Spectral')
	C = c(np.array(np.round(np.linspace(0, 255, Y.shape[0])), dtype=np.int64))
	C = C[:, 0:3]
	plt.figure(figsize=(15, 6))
	ax = plt.subplot(131)
	ax.plot(x)
	ax.set_title("Original Signal")
	ax.set_xlabel("Phase")
	ax2 = plt.subplot(132)
	ax2.set_title("PCA of Sliding Window Embedding")
	ax2.scatter(Y[:, 0], Y[:, 1], c=C)
	ax2.set_aspect('equal', 'datalim')
	ax3 = plt.subplot(133)
	I = PDs[1]
	ax3.set_title("Max Persistence = %.3g"%np.max(I[:, 1] - I[:, 0]))
	plotDGM(I)
	plt.savefig(filename)
	plt.close()


analyze_SP500(num_days=100)

