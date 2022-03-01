# knn-algorithm-from-scratch
K-Nearest-Neighbor classification and regression algorithm written by hand

My first machine learning algorithm, well really my first coding project ever. Diving into this head first and loving every second of it :)

Tested on sklearn's Iris data, as well as a dataset on fish found via Kaggle. https://www.kaggle.com/aungpyaeap/fish-market

File contains functions to calculate Euclidean and Manhattan distance
These are used to calculate a test point(s) k nearest neighbors

Returning mode for classification, -1 when a tie. And the mean for regression

A function is also included to cross validate the function using folds to test/train split the data and return an accuracy score
Also includes pair plots for both datasets and the ability to print confusion matrices when evaluating.
