# Homework 5 for Cynthia Rudin's Machine Learning class
# All State challenge from Kaggle
# Jeremy Fox

import numpy as np
import pandas as pd
from sklearn import metrics, preprocessing
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.metrics import explained_variance_score, mean_absolute_error, r2_score
from sklearn.linear_model import RidgeCV, LinearRegression, LassoCV
from sklearn.kernel_ridge import KernelRidge
from sklearn.neural_network import MLPRegressor
import csv

# Remove ids from a dataframe and return the array of ids
def strip_ids(df):
    ids = df['id']
    del df['id']
    return ids
# Remove both ids and Y values from a dataframe and return a tuple of the arrays, where
# ids comes first, and Y values come second
def strip_Y_ids(df):
    ids = df['id']
    del df['id']
    Y = df['loss'].tolist()
    del df['loss']
    return (ids, Y)

# Read in stock data
def read_data(filename):
    # Reads in data, using headers for column names and inferring the type
    df = pd.read_csv(filename)

    # delete id column and save and delete Y values
    # ids, Y = strip_Y_ids(df)


    # convert dataframe to matrix
    X = df.as_matrix()

    # drop features based on mutual information
    # selector = SelectKBest(mutual_info_regression, k=100)
    # selector.fit(X,Y)
    # selector.transform(X)

    # Here, X is the training data, Y is the labels, les is a set of LabelEncoders,
    # enc is the OneHotEncoder, and selector is the feature selector
    return X

# returns a list of tuples of a datapoint and its label
def combine_data_and_labels(data, labels):
    return zip(data,labels)

# writes out a matrix as a csv at given file path. Uses a comma as delimiter
def write_csv(matrix, filepath):
    ofile = open(filepath,"wb")
    writer = csv.writer(ofile,delimiter=',')
    for row in matrix:
        writer.writerow(row)
    ofile.close()

# Begin main code body

# Read in feature vectors
X = read_data("sample_vectors.csv")

# Read in indicator values
y_col = read_data("sample_indicators.csv")
y = np.ravel(y_col)

# Set up to use different algorithms

# # Set up MLPRegressor
# mlp_parameters = {'hidden_layer_sizes': [(4)], 'activation': ['identity', 'logistic', 'tanh']}
# mlp_algo = GridSearchCV(MLPRegressor(max_iter=1000), mlp_parameters, cv=5)

# Set up Linear Regression
linear_algo = LinearRegression()

# Set up Ridge Regression
ridge_alphas = np.array([0.1, 1, 10, 100, 1000])
ridge_algo = RidgeCV(alphas=ridge_alphas, fit_intercept=True)

# Set up LASSO
lasso_alphas = np.array([0.1, 1, 10, 100, 1000])
lasso_algo = LassoCV(alphas=lasso_alphas, fit_intercept=True)

algos = [(linear_algo, "Linear Regression"), (ridge_algo, "Ridge"), (lasso_algo, "Lasso")]

# Use cross validation to prevent overfitting
for pair in algos:
    algo = pair[0]
    label = pair[1]
    print("Computing cross validation scores...")
    scores = cross_val_score(algo, X, y, cv=5, scoring='neg_mean_absolute_error')
    print("Printing scores for " + label)
    for score in scores:
        print(score)

# Split the dataset into two equal parts
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Do actual training with the different algorithms compute metrics, and write out a csv
for pair in algos:
    algo = pair[0]
    label = pair[1]
    # fit and predict
    algo.fit(X_train, y_train)
    y_true, y_pred = y_test, algo.predict(X_test)

    print(label)
    # print out weights
    print("Weights: ", algo.coef_)
    print("Intercept: ", algo.intercept_)

    # Print out metrics
    print(explained_variance_score(y_true, y_pred))
    print(mean_absolute_error(y_true, y_pred))
    print(r2_score(y_true, y_pred))

print("finished")
