#imports
import numpy as _numpy
import matplotlib.pyplot as _mathplotlib
import pandas as _pandas

data = _pandas.read_csv(r"C:\\NICOLAS\\Curso MachineLearning with Python\\datasets\\distributed-data\\001.csv")
data.shape

# Importar el primer fichero
# hacemos un bucle para ir recorriendo c/u de los ficheros.
# Es importante tener una consistencia en el nombre de los ficheros
# Una vez dentro del bucle, c/u debe apendizarse (añadirse al final)
# del primer fichero que ya habiamos cargado.
# Repito el bucle hasta que no queden ficheros.

filepath = r"C:\\NICOLAS\\Curso MachineLearning with Python\\datasets\\distributed-data"
data = _pandas.read_csv(r"C:\\NICOLAS\\Curso MachineLearning with Python\\datasets\\distributed-data\\001.csv")
final_length = len(data)

for i in range(2, 333):
    if (i < 10):
        filename = "00"+ str(i)
    if (10 <= i <= 100):
        filename = "0"+str(i)
    if (i >= 100):
        filename = str(i)
    
    file = filepath + "\\" + filename + ".csv"
    temp_data = _pandas.read_csv(file)
    final_length += len(temp_data)
    
    data = _pandas.concat([data, temp_data], axis=0)
    data.shape

# corroboro si se cargaron todos los csv
final_length == data.shape[0]









