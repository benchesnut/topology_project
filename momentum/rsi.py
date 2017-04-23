import pandas as pd
import numpy as np
import stockstats as ss

def RSI(x):

    # n = len(x)
    # delta = np.zeros(n-1)
    # for a in range(1, n):
    #     delta[a-1] = x[a] - x[a-1]
    #
    # dUp, dDown = delta.copy(), delta.copy()
    # for n in range(len(dUp)):
    #     if dUp[n] < 0:
    #         dUp[n] = 0
    #         dDown[n] = -dDown[n]
    #     else:
    #         dDown[n] = 0
    #
    # RolUp = pd.rolling_mean(dUp, n)
    # RolDown = pd.rolling_mean(dDown, n)
    #
    # if RolDown == 0:
    #     return 100
    #
    # RS = RolUp/RolDown
    # return 100-(100/(1+RS))
