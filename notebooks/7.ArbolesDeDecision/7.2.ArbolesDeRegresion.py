# ÁRBOLES DE REGRESIÓN

# Imports
import pandas as _pandas
import matplotlib.pyplot as _plt
from sklearn.tree import DecisionTreeRegressor

filepath = r"C:\\Nicolas\\Curso MachineLearning with Python\\datasets\\boston"
file = filepath + "\\" + "Boston.csv"
data = _pandas.read_csv(file)
data

# crim = indice de crimonología per capita
# zn = proporcion de residentes por cada 25 mil metros cuadrados
# para el resto de variables, buscar en Google el dataset
# de Housing Values on Suburbs of Boston.

colnames = data.columns.values.tolist()
# uso todas las columnas como predictoras menos la ultima
predictors = colnames[:13]
target = colnames[13]
X = data[predictors]
Y = data[target]

# armo el árbol ->
regtree = DecisionTreeRegressor(min_samples_split=30, min_samples_leaf=10, max_depth=5, random_state=0)
regtree.fit(X,Y)
# hago predicciones ->
preds = regtree.predict(data[predictors])
# agrego las predicciones al data ->
data["preds"] = preds
data[["preds", "medv"]]

from sklearn.tree import export_graphviz
with open(r"C:\\Nicolas\\Curso MachineLearning with Python\\notebooks\\Recursos\\reg_tree.dot", "w") as dotfile:
    export_graphviz(regtree, out_file=dotfile, feature_names=predictors)
    dotfile.close()
    
import os
from graphviz import Source
file = open("resources/boston_rtree.dot", "r")
text = file.read()
Source(text)

# validacion cruzada para ver mejor la eficacia del modelo ->
from sklearn.cross_validation import KFold
from sklearn.cross_validation import cross_val_score
import numpy as np

cv = KFold(n=X.shape[0], n_folds = 10, shuffle=True, random_state=1)
scores = cross_val_score(regtree, X, Y, scoring="mean_squared_error", cv = cv, n_jobs=1)
print(scores)
score = np.mean(scores)
print(score)

list(zip(predictors,regtree.feature_importances_))


# RANDOM FOREST
# (la ventaja del RANDOM FOREST es que no necesita validación cruzada).
# Utiliza el método de Bagging...detrás, de fondo está la Ley de los Grandes Números.
# Además, en el bosque aleatorio, no hay poda de árboles.
from sklearn.ensemble import RandomForestRegressor

forest = RandomForestRegressor(n_jobs=2, oob_score=True, n_estimators=10000)
forest.fit(X,Y)

data["rforest_pred"]= forest.oob_prediction_
data[["rforest_pred", "medv"]]

data["rforest_error2"] = (data["rforest_pred"]-data["medv"])**2
sum(data["rforest_error2"])/len(data)

forest.oob_score_














