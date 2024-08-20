# Algoritmo K-Nearest casero
# basado en distancia euclídea

# imports
import numpy as _numpy
import matplotlib.pyplot as _plt
from matplotlib import style
import warnings
from math import sqrt
from collections import Counter
import pandas as _pandas

# defino mi dataset
dataset = {
    'k': [[1,2],[2,3],[3,1]], # k = color negro
    'r': [[6,5],[7,7],[8,6]] # r = color rojo
    }
# nuevo punto a clasificar ->
new_point = [5,7]

# plot
# manera 1
[[_plt.scatter(ii[0],ii[1], s=50, color = i) for ii in dataset[i]] for i in dataset]
_plt.scatter(new_point[0],new_point[1], s = 100)
# manera 2
#for i in dataset:
#    for ii in dataset[i]:
#        _plt.scatter(ii[0],ii[1], s=50, color = i)
#        _plt.scatter(new_point[0],new_point[1], s = 100)

def k_nearest_neighbors(data,predict, k=3, verbose = False):
    if (len(data) >= k):
        warnings.warn("K es un valor menor que el número total de elementos a votar!!")
    
    # calculo la distancia euclídea (raiz cuadrada de la distancia... -1 valor a prediecir al cuadrado... etc)
    distances = []
    for group in data:
        for feature in data[group]:
            # MANERA 1
            #d_euclidea = sqrt((feature[0]-predict[0])**"" + (feature[1]-predict[1])**2)
            # MANERA 2
            #d_euclidea = _numpy.sqrt(_numpy.sum((_numpy.array(feature) - _numpy.array(predict))**2))
            # MANERA 3 --> calculando la norma de un vector.
            d_euclidea = _numpy.linalg.norm(_numpy.array(feature)- _numpy.array(predict))
            distances.append([d_euclidea,group])
    if (verbose):
        print(distances)
    # me agarro todas las distancias de la 0 a la k-ésima.
    # el sorted lo que hace es ordenar por la primera columna proporcionada.
    votes = [i[1] for i in sorted(distances)[:k]]
    if (verbose):
        print(votes)
    vote_result = Counter(votes).most_common(1)
    if (verbose):
        print(vote_result)
    
    return vote_result[0][0]

# llamo a la función
new_point = [4.4,4.5]
result = k_nearest_neighbors(dataset, new_point)
result

[[_plt.scatter(ii[0],ii[1], s=50, color = i) for ii in dataset[i]] for i in dataset]
_plt.scatter(new_point[0],new_point[1], s = 100)

## Aplico el modelo KNN al dataset del Cancer
df = _pandas.read_csv(r"C:\NICOLAS\Curso MachineLearning with Python\datasets\cancer\breast-cancer-wisconsin.data.txt")
# inserto en df un replace para los valores nulls o desconocidos ->
df.replace("?",-99999, inplace=True)

# limpio las columnas ->
df.columns = ["name", "V1", "V2","V3","V4","V5","V6","V7","V8","V9","class"]
df = df.drop(["name"],1)

full_data = df.astype(float).values.tolist()
full_data

import random
# ordeno los datos aleatoriamente
random.shuffle(full_data)
test_size = 0.2
train_set = {2:[],4:[]}
test_set = {2:[],4:[]}
# para entrenar al modelo agarro todos los elementos menos el ultimo 20% ->
train_data = full_data[:-int(test_size*len(full_data))]
# para testear el modelo, agarro todos los elementos, desde el ultimo 20% hasta el final ->
test_data = full_data[-int(test_size*len(full_data)):]

# asigno valores a los conjuntos
for i in train_data:
    train_set[i[-1]].append(i[:-1])
for i in test_data:
    test_set[i[-1]].append(i[:-1])

train_set

correct = 0
total = 0
for group in test_set:
    for data in test_set[group]:
        vote = k_nearest_neighbors(train_set, data, k = 5)
        if (group == vote):
            correct += 1
        total += 1
print("Eficacia del KNN =",correct/total)
## FIN Aplico el modelo KNN al dataset del Cancer





























