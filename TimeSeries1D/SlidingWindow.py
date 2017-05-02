import numpy as np
import scipy.interpolate as interp
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# import TDA
import quandl

quandl.ApiConfig.api_key = "DqLVWRStVw_hyQnnQvyW"

def getSlidingWindow(x, dim, Tau, dT):
    import ipdb; ipdb.set_trace()
    N = len(x)
    NWindows = int(np.floor((N-dim*Tau)/dT))
    X = np.zeros((NWindows, dim))
    print(X)
    idx = np.arange(N)
    for i in range(NWindows):
        idxx = dT*i + Tau*np.arange(dim)
        start = int(np.floor(idxx[0]))
        end = int(np.ceil(idxx[-1]))+2
        if end >= len(x):
            X = X[0:i, :]
            break
        X[i, :] = interp.spline(idx[start:end+1], x[start:end+1], idxx)
        print(X[i, :])
    return X

def getSlidingWindowNoInterp(x, dim):
    N = len(x)
    NWindows = N - dim + 1
    X = np.zeros((NWindows, dim))
    idx = np.arange(N)
    for i in range(NWindows):
        X[i, :] = x[i:i+dim]
    return X

def normalizeWindows(X):
    for i in range(0, len(X)):
        first = X[i][0]
        for j in range(0, len(X[i])):
            X[i][j] = (X[i][j] - first) / first
    return X
      

if __name__ == '__main__':
    # import ipdb; ipdb.set_trace()
    T = 50 #The period in number of samples
    NPeriods = 20 #How many periods to go through
    N = T*NPeriods #The total number of samples
    t = np.linspace(0, 2*np.pi*NPeriods, N+1)[0:N] #Sampling indices in time
    x = np.cos(t) #The final signal
    # data = quandl.get("GOOG/NASDAQ_GOOG", returns="numpy", rows=250)
    # x = [y[1] for y in data]
    
    # prev_day = x[99]
    # for i in range(100, 900):
    #     new_price = prev_day + np.random.uniform(-.29, .31)
    #     x[i] = new_price
    #     prev_day = x[i]

    # for i in range(900, 1000):
    #     x[i] = x[i] + prev_day

    dim = 10
    Tau = 0.5
    dT = 0.1
    X = getSlidingWindowNoInterp(x, dim)
    X = normalizeWindows(X)
    extent = Tau*dim
    PDs = doRipsFiltration(X, 1)
    pca = PCA(n_components = 2)
    Y = pca.fit_transform(X)
    eigs = pca.explained_variance_

#    for i in range(X.shape[0]):
#        plt.clf()
#        idxx = dT*i + Tau*np.arange(dim)
#        plt.stem(idxx, X[i, :], 'r')
#        plt.hold(True)
#        start = int(np.floor(idxx[0]))
#        end = int(np.ceil(idxx[-1]))
#        plt.plot(start + np.arange(end-start+1), x[start:end+1])
#        plt.savefig("Window%i.png"%i)

    # Step 4: Plot original signal and PCA of the embedding
    c = plt.get_cmap('Spectral')
    C = c(np.array(np.round(np.linspace(0, 255, Y.shape[0])), dtype=np.int64))
    C = C[:, 0:3]
    plt.figure(figsize=(15, 6))
    ax = plt.subplot(131)
    ax.plot(x)
    # ax.set_ylim((700, 900))
    ax.set_title("Original Signal")
    ax.set_xlabel("Phase")
    ax2 = plt.subplot(132)
    ax2.set_title("PCA of Sliding Window Embedding")
    ax2.scatter(Y[:, 0], Y[:, 1], c=C)
    ax2.set_aspect('equal', 'datalim')
    ax3 = plt.subplot(133)
    I = PDs[1]
    # ax3.set_title("Max Persistence = %.3g"%np.max(I[:, 1] - I[:, 0]))
    plotDGM(I)
    # plt.savefig('Google1.png')
    plt.show()

    # #Step 5: Plot original signal and the persistence diagram
    # fig = plt.figure(figsize=(12, 6))
    # ax = plt.subplot(121)
    # ax.plot(x)
    # ax.set_ylim((700, 900))
    # ax.set_title("Original Signal")
    # ax.set_xlabel("Sample Number")
    # ax.hold(True)
    # ax.plot([extent, extent], [np.min(x), np.max(x)], 'r')

    # ax2 = fig.add_subplot(122)
    # I = PDs[1]
    # plotDGM(I)
    # plt.title("Max Persistence = %.3g"%np.max(I[:, 1] - I[:, 0]))
    # plt.savefig('Google2.png')
    # # plt.show()
