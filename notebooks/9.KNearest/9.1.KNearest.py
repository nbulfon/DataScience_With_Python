# imports
import numpy as _numpy
from sklearn import preprocessing, model_selection, neighbors
import pandas as _pandas

df = _pandas.read_csv(r"C:\NICOLAS\Curso MachineLearning with Python\datasets\cancer\breast-cancer-wisconsin.data.txt")
# limpio las columnas ->
df.columns = ["name", "V1", "V2","V3","V4","V5","V6","V7","V8","V9","class"]
df.head()

df = df.drop(["name"],1)
df.head()

# inserto en df un replace para los valores nulls o desconocidos ->
df.replace("?",-99999, inplace=True)

# seteo mi var. explidada y las explicativas ->
Y = df["class"]
X = df[["V1", "V2","V3","V4","V5","V6","V7","V8","V9"]]
X.head()


# Clasificación de los K vecinos (Fase entrenamiento y prueba)

# armo conjunto entrenamiento y de test ->
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X,Y,test_size=0.2)
# creo el modelo
clf = neighbors.KNeighborsClassifier()
# hago un fit para entrenar al modelo
clf.fit(X_train, Y_train)

# corroboro la precisión del modelo que armé ->
accuracy = clf.score(X_test, Y_test)
accuracy

## Clasifcar nuevos datos

sample_measure = _numpy.array([4,2,1,1,1,2,3,2,1])
sample_measure = sample_measure.reshape(1,-1)
predict = clf.predict(sample_measure)
predict
sample_measure2 = _numpy.array([4,2,1,1,1,2,3,2,1],[4,2,1,1,1,2,3,2,1]).reshape(2,-1)
predict = clf.predict(sample_measure2)
predict
## FIN Clasificar nuevos datos


# FIN Clasificación de los K vecinos


### Clasificación SIN LIMPIEZA
# (test para demostrar como la clasificación sin limpieza 
#(sin data cleaning) no sirve de nada si se quiere armar
# un buen modelo)
df = _pandas.read_csv(r"C:\NICOLAS\Curso MachineLearning with Python\datasets\cancer\breast-cancer-wisconsin.data.txt")
df.replace("?",-99999, inplace=True)
df.columns = ["name", "V1", "V2","V3","V4","V5","V6","V7","V8","V9","class"]
Y = df["class"]
X = df[["name","V1", "V2","V3","V4","V5","V6","V7","V8","V9"]]
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X,Y,test_size=0.2)
clf = neighbors.KNeighborsClassifier()
clf.fit(X_train, Y_train)
accuracy_sin_limpieza = clf.score(X_test, Y_test)
accuracy_sin_limpieza
 # notar como baja la eficacia (accuracy_sin_limpieza < accuracy)
# al añadir una variable redundante o que no importa (columna 'name').
### FIN Clasificación SIN LIMPIEZA








