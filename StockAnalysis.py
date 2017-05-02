import csv
import quandl
import sys, os
import TimeSeries1D.TDA as TDA
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import filtration
import momentum.rsi as rsi
# import stock_data_model

from TimeSeries1D import SlidingWindow


quandl.ApiConfig.api_key = "DqLVWRStVw_hyQnnQvyW"

# dim is the length of each window, default 10
def analyze_SP500(dim=10, num_days=100):
	# use for analyzing entire sp500
	sp500 = open('TimeSeries1D/SP500.csv', 'rt')
	print(sp500)
	pd_map = {}
	input_file = open('input_vectors.csv', "w")
	output_file = open('output_vectors.csv',"w")
	in_writer = csv.writer(input_file,delimiter=',')
	out_writer = csv.writer(output_file,delimiter=',')

	# inputs, outputs = get_data_stock('ADBE', in_writer, out_writer, num_days)

	for row in csv.reader(sp500, delimiter=','):
		get_data_stock(row[0], in_writer, out_writer, num_days, dim)        	
		# if PDs is not None:
		# 	make_plot(row[0], PDs)
		# 	pd_map[row[0]] = PDs[1]

def calc_input(ticker, data, x):
	temp = filtration.filter(data)
	p_points = [(temp[j][0], temp[j][1]-temp[j][0]) for j in range(len(temp))]
	# print(p_points)
	# plt.scatter(*zip(*p_points))
	# plt.show()
	# points = [(j, x[j]) for j in range(len(x))]
	# plt.plot(*zip(*points))
	# plt.show()

	input_vector = TDA.calcBins(p_points, 20, 20)
	return input_vector

def calc_output(ticker, prices):

	return rsi.relative_strength(prices, n=14)

def get_data_stock(ticker, in_writer, out_writer, num_days=3000, dim=100):
	# gather the data
	try:
		data = quandl.get("GOOG/NASDAQ_" + ticker, returns="numpy", rows=num_days)
	except quandl.errors.quandl_error.NotFoundError:
		print(ticker)
		return
		# return None
	x = [y[1] for y in data]

	# get the sliding window vectors
	X = SlidingWindow.getSlidingWindowNoInterp(x, dim)

	X_norm = normalizeWindows(X)

	# do TDA and PCA
	input_vectors = []
	outputs = []
	for i in range(num_days - dim):
		# Calc input
		input_vectors.append(calc_input(ticker, X_norm[i], x))
		in_writer.writerow(input_vectors[i])
		#Calc output
		outputs.append(calc_output(ticker, X[i]))
		out_writer.writerow([outputs[i]])
	
	# make_plot(ticker, PDs, X_norm, x)
	# return input_vectors, outputs

def normalizeWindows(X):
	for i in range(0, len(X)):
		first = X[i][0]
		for j in range(0, len(X[i])):
			X[i][j] = X[i][j] / first
	return X
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

def make_plot(ticker, PDs, X, x):
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


analyze_SP500(dim=100, num_days=500)
