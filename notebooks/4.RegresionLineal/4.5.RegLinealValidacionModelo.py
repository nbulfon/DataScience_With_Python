# imports
import pandas as _pandas
import statsmodels.formula.api as _smf
import matplotlib.pyplot as _plt
import numpy as _numpy

filepath = r"C:\\Nicolas\\Curso MachineLearning with Python\\datasets\\ads"
file = filepath + "\\" + "Advertising.csv"
data = _pandas.read_csv(file)

## Dividir el dataset en conjunto de entrenamiento y de testing
## Esto se debe hacer al principio del analisis...para no ensuciar los conjuntos entre sí.

conjunto_a = _numpy.random.randn(len(data))
_plt.hist(conjunto_a)

# seteo la condicion que quiero, así separo mis dos conjuntos ->
check = (conjunto_a < 0.8)
conjunto_training = data[check]
conjunto_testing = data[~check]

len(conjunto_training), len(conjunto_testing)

# armo mi modelo lineal (training)
linearModel = _smf.ols(formula="Sales~TV+Radio", data=conjunto_training).fit()
linearModel.summary()

## Validacion del modelo con el conjunto de testing.
sales_predict = linearModel.predict(conjunto_testing)
sales_predict

SSD = sum( (conjunto_testing["Sales"] - sales_predict) **2)
SSD

RSE = _numpy.sqrt(SSD / len(conjunto_testing)-2-1)
RSE

sales_mean = _numpy.mean(conjunto_testing["Sales"])
error = RSE / sales_mean
error



