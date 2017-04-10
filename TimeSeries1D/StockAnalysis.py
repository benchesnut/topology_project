import csv

import quandl

from SlidingWindow import *

quandl.ApiConfig.api_key = "DqLVWRStVw_hyQnnQvyW"

# dim is the length of each window, default 10


def analyze_SP500(dim=10, num_days=100):
	# use for analyzing entire sp500
	sp500 = open('SP500.csv', 'rt')
	for row in csv.reader(sp500, delimiter=','):
		analyze_stock(row[0], num_days=num_days)


def analyze_stock(ticker, dim=10, num_days=100):
	# gather the data
	try:
		data = quandl.get("GOOG/NASDAQ_" + ticker, returns="numpy", rows=100)
	except quandl.errors.quandl_error.NotFoundError:
		print(ticker)
		return
	x = [y[1] for y in data]
    	
	# get the sliding window vectors
	X = getSlidingWindowNoInterp(x, dim)

	# do TDA and PCA
	PDs = doRipsFiltration(X, 2)
	pca = PCA(n_components = 2)
	Y = pca.fit_transform(X)
	eigs = pca.explained_variance_

	make_plot(ticker, PDs, Y, x)

def make_plot(ticker, PDs, Y, x):
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

