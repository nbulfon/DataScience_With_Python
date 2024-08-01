#imports
import numpy as _numpy
import matplotlib.pyplot as _mathplotlib
import pandas as _pandas

red_wine = _pandas.read_csv(r"C:\NICOLAS\Curso MachineLearning with Python\datasets\wine\winequality-red.csv" ,sep=";")
white_wine = _pandas.read_csv(r"C:\NICOLAS\Curso MachineLearning with Python\datasets\wine\winequality-white.csv", sep=";")

print(red_wine.columns)
# veo la cantidad de registros que hay ->
print(white_wine.shape)

# En Python, tenemos dos tipos de ejes.
# axis = 0 denota el eje horizontal
# axis = 1 denota el eje vertical

# acá notar que los concateno enteramente de manera horizontal porque justo son
# dos conjuntos de datos con todas las columnas identicas. No hace falta ir columna x columna.s
wine_data = _pandas.concat([red_wine, white_wine], axis= 0)
print(wine_data.shape)

# pruebas
data1 = wine_data.head(10)
data2 = wine_data[300:310]
data3 = wine_data.tail(10)

# los combino (screamble)
wine_scramble = _pandas.concat([data1,data2,data3], axis=0)
print(wine_scramble)