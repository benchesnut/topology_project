import numpy as np
import pandas as pd

def ROC(x):
    return ((x[len(x)-1] - x[0]) / x[0]) * 100
