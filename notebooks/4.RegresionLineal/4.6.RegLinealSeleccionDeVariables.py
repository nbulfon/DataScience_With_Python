# imports
import pandas as _pandas
import statsmodels.formula.api as _smf
import matplotlib.pyplot as _plt
import numpy as _numpy
from sklearn.feature_selection import RFE
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression

filepath = r"C:\\Nicolas\\Curso MachineLearning with Python\\datasets\\ads"
file = filepath + "\\" + "Advertising.csv"
data = _pandas.read_csv(file)

# nombres de las columnas que seran variables predictivas (explicativas)
feature_cols = ["TV", "Radio", "Newspaper"]
X = data[feature_cols]
Y = data["Sales"]

estimator = SVR(kernel="linear")
selector = RFE(estimator=estimator, n_features_to_select=2, step=1)
selector = selector.fit(X, Y)

print(selector.support_)
print(selector.ranking_)

X_pred = X[["TV", "Radio"]]
linearRegression = LinearRegression()
linearRegression.fit(X_pred, Y)
linearRegression.intercept_
linearRegression.coef_
