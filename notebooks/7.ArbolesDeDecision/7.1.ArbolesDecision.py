# ÁRBOL DE DECISION PARA ESPECIES DE FLORES

# Imports
import pandas as _pandas
import matplotlib.pyplot as _plt
import numpy as _numpy
from sklearn.tree import DecisionTreeClassifier

filepath = r"C:\\Nicolas\\Curso MachineLearning with Python\\datasets\\iris"
file = filepath + "\\" + "iris.csv"
data = _pandas.read_csv(file)
data

_plt.hist(data.Species)

# las 4 primeras son predictoras y la 4ta es la var. objetivo
colnames = data.columns.values.tolist()
predictors = colnames[:4]
target  = colnames[4]

data["is_train"] = _numpy.random.uniform(0,1,len(data))<=0.75

# defino las dos variables de una ->
train,test = data[data["is_train"]==True], data[data["is_train"]==False]

# creo el árbol
tree = DecisionTreeClassifier(criterion="entropy"
                              ,min_samples_split=20,
                              random_state=99)
tree.fit(train[predictors], train[target])

# seteo las predicciones
preds = tree.predict(test[predictors])

_pandas.crosstab(test[target],
                 preds,
                 rownames=["Actual"],
                 colnames=["Predictions"])

## Visualización del árbol de decisión
from sklearn.tree import export_graphviz

# indico dónde voy a guardar el fichero ->
path_tree = r"C:\\Nicolas\\Curso MachineLearning with Python\\notebooks\\Recursos\\iris_dtree.dot"
with open(path_tree,"w") as dotfile:
    export_graphviz(tree,out_file=dotfile,feature_names=predictors)
    dotfile.close()
    
import os
from graphviz import Source

archivoArbol = open(filepath)
text = archivoArbol.read()
text

Source(text)
## FIN Visualización del árbol de decisión

## Cross-Validation para la poda

# creo un nuevo árbol de decisión ->
X = data[predictors]
Y = data[target]

tree = DecisionTreeClassifier(criterion="entropy",
                              max_depth=5,
                              min_samples_split=20,
                              random_state=99)
tree.fit(X,Y)

# implemento validación cruzada
from sklearn.cross_validation import KFold

cv = KFold(n = X.shape[0], n_folds=10, shuffle=True, random_state=1)

from sklearn.cross_validation import cross_val_score

# calculo el promedio de las eficacias en cada division hecha ->
scores = cross_val_score(tree, X, Y, scoring="accuracy", cv = cv, n_jobs=1)
scores

score = _numpy.mean(scores)
score

for i in range(1,11):
    tree = DecisionTreeClassifier(criterion="entropy", max_depth=i, min_samples_split=20, random_state=99)
    tree.fit(X,Y)
    cv = KFold(n = X.shape[0], n_folds=10, shuffle=True, random_state=1)
    scores = cross_val_score(tree, X, Y, scoring="accuracy", cv = cv, n_jobs=1)
    score = _numpy.mean(scores)
    print("Score para i = ",i," es de ", score)
    print("   ",tree.feature_importances_)

predictors

#Random forest

from sklearn.ensemble import RandomForestClassifier

forest = RandomForestClassifier(n_jobs=2, oob_score=True, n_estimators=100)
forest.fit(X,Y)

forest.oob_decision_function_

forest.oob_score_



## FIN Cross-Validation para la poda
























