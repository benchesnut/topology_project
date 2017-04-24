import numpy as np
import scipy.ndimage.filters as fi
import matplotlib.pyplot as plt

def gkern2(kernlen=21, nsig=3):
    """Returns a 2D Gaussian kernel array."""

    # create nxn zeros
    inp = np.zeros((kernlen, kernlen))
    # set element at the middle to one, a dirac delta
    inp[kernlen//2, kernlen//2] = 1
    # gaussian-smooth the dirac, resulting in a gaussian filter mask
    return fi.gaussian_filter(inp, nsig)


# MAIN BODY
gauss = gkern2(75, 25)
matrix = np.zeros((3, 3))
for i in range(3):
    for j in range(3):
        for k in range(25):
            for l in range(25):
                matrix[i][j] += gauss[25*i + k][25*j + l]
print(matrix)
print(gkern2(3, 1))
