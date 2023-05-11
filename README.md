# knn-algorithm-from-scratch
K-Nearest-Neighbor classification and regression algorithm written by hand.

My first machine learning algorithm -

Tested on sklearn's Iris data, & a dataset found via Kaggle on fish. https://www.kaggle.com/aungpyaeap/fish-market

File contains functions to calculate Euclidean and Manhattan distance.
These are used to calculate a test point's k nearest neighbors

Returns mode for classification, -1 when a tie.
Returns mean for regression.

A function is also included to cross validate using folds to test/train split the data and return an accuracy score.
Also includes pair plots for both datasets and the ability to print confusion matrices when evaluating.
