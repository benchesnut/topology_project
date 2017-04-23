import numpy as np
import pandas as pd
import rsi
import roc

def read_data(filename):
    df = pd.read_csv(filename)
    X = df.as_matrix()
    return X



### MAIN BODY
df = read_data("../sample_vectors.csv")
for row in df:
    print(roc.ROC(row))
