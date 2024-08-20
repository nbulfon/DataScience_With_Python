import pandas as _pandas
import numpy as _numpy

# carga de datos
df = _pandas.read_csv(
    r"C:\NICOLAS\Curso MachineLearning with Python\datasets\ml-100k\u.data.csv",
    sep="\t",
    header=None)
df.shape
# data wrangling
df.columns = ["UserID","ItemID","Raiting","TimeStamp"]

# análisis exploratorio de los ítems.
import matplotlib.pyplot as _plt
_plt.hist(df.Raiting)
_plt.hist(df.TimeStamp)
df.groupby(["Raiting"])["UserID"].count()

_plt.hist(df.groupby(["ItemID"])["ItemID"].count())

# Representación en forma matricial

numeroUsuarios = df.UserID.unique().shape[0]
numeroUsuarios

numeroItems = df.ItemID.unique().shape[0]
numeroItems

# armo una mtriz de valoraciones, en la que las filas seran usuarios,
# y las columnas serán peliculas.

ratings = _numpy.zeros((numeroUsuarios,numeroItems))
ratings

# transformo todo el Df en una matriz n-dimensional
for row in df.itertuples():
    ratings[row[1]-1,row[2]-1] = row[3]

# veo que % de la matriz no tiene ceros
sparsity = float(len(ratings.nonzero()[0]))
sparsity /= (ratings.shape[0]*ratings.shape[1])
sparsity *= 100
print("Coeficiente de sparseidad: {:4.2f}%".format(sparsity))

# Creo conjuntos de entrenamiento y validación para mi modelo
from sklearn.model_selection import train_test_split

ratings_train, ratings_test = train_test_split(ratings, test_size = 0.3, random_state=42)
ratings_train.shape

# Filtro colaborativo basado en Usuarios 
## Matriz de similaridad entre los usuarios (distancia del coseno // 0<cos<1).
## Predecir la valoración desconocida de un item *i* para un usuario activo *u*
## basandonos en la suma ponderada de todas las valoraciones del resto de
## usuarios para dicho ítem.
## Con todo esto, recomendaremos los nuevos ítems a los usuarios, según lo 
# establecido en los pasos anteriores.

import numpy as _numpy
import sklearn

# creo matriz de similaridad
sim_matrix = 1 - sklearn.metrics.pairwise.cosine_distances(ratings_train)
type(sim_matrix)
sim_matrix.shape

# hago predicciones
normalization_factor = _numpy.abs(sim_matrix).sum(axis=1).reshape(-1, 1)
users_predictions = sim_matrix.dot(ratings_train) / _numpy.array([_numpy.abs(sim_matrix).sum(axis=1)]).T
users_predictions



print("Shape of users_predictions:", users_predictions.shape)
print("Shape of raitings_train:", ratings_train.shape)
from sklearn.metrics import mean_squared_error

# obtener el error cuadrático.
def get_mse(preds, actuals):
    if preds.shape[0] != actuals.shape[0]:
        actuals = actuals.T
    preds = preds[actuals.nonzero()].flatten()
    actuals = actuals[actuals.nonzero()].flatten()
    return mean_squared_error(preds, actuals)

get_mse(users_predictions, ratings_train)

sim_matrix = 1 - sklearn.metrics.pairwise.cosine_distances(ratings_test)
users_predictions = sim_matrix.dot(ratings_test) / _numpy.array([_numpy.abs(sim_matrix).sum(axis=1)]).T
get_mse(users_predictions, ratings_test)


# Filtro colaborativo basado en los KNN

from sklearn.neighbors import NearestNeighbors

k = 5
# le paso a NearestNeigboors, la cantidad (k) y el tipo de distancia (coseno).
neighbors = NearestNeighbors(n_neighbors=k, metric='cosine')
neighbors.fit(ratings_train)

top_k_distances, top_k_users = neighbors.kneighbors(ratings_train, return_distance=True)

top_k_distances.shape
top_k_distances[0]
top_k_users.shape
top_k_users[0]

# hago la prediccion
users_predicts_k = _numpy.zeros(ratings_train.shape)
# para cada usuario del conjunto de entrenamiento ... (luego normalizo)
# Corrigiendo la indexación en el bucle
for i in range(ratings_train.shape[0]):
    users_predicts_k[i,:] = top_k_distances[i].T.dot(ratings_train[top_k_users[i]]) / _numpy.array([_numpy.abs(top_k_distances[i].T).sum(axis=0)]).T


users_predicts_k.shape    

# veo si el modelo mejora o no ->
get_mse(users_predicts_k, ratings_train)

users_predicts_k = _numpy.zeros(ratings_test.shape)
for i in range(ratings_test.shape[0]):# para cada usuario del conjunto de test
    users_predicts_k[i,:] = top_k_distances[i].T.dot(ratings_test[top_k_users][i]) / _numpy.array([_numpy.abs(top_k_distances[i].T).sum(axis=0)]).T
get_mse(users_predicts_k, ratings_test)
ratings_test
# FIN Filtro colaborativo basado en los KNN

# Filtro colaborativo basado en Items
# es como el filtro basado en usuarios, pero en los items
# (en este caso, películas).
n_movies = ratings_train.shape[1]
n_movies
#neighboors = NearestNeighbors(n_neighbors=n_movies, metric='cosine')
neighbors = NearestNeighbors(n_neighbors=n_movies, metric='cosine')

neighbors.fit(ratings_train.T)



top_k_distances, top_k_items = neighbors.kneighbors(ratings_train.T, return_distance=True)
top_k_items

# prediccion de items ->
item_preds = ratings_train.dot(top_k_distances) / _numpy.array([_numpy.abs(top_k_distances).sum(axis=1)])
item_preds.shape

# obtengo el error cuadrático del conjunto de entrenamiento ->
get_mse(item_preds, ratings_train)

# obtengo el error cuadrático pero del conjunto de test ->
get_mse(item_preds, ratings_test)


df = _pandas.read_csv(
    r"C:\NICOLAS\Curso MachineLearning with Python\datasets\ml-100k\u.data.csv",
    sep="\t",
    header=None)

# FIN Filtro colaborativo basado en Items

# Filtro colaborativo basado en los KNN

k = 30
neighbors = NearestNeighbors(n_neighbors=k, metric='cosine')
neighbors.fit(ratings_train.T)
top_k_distances, top_k_items = neighbors.kneighbors(ratings_train.T, return_distance=True)

top_k_distances.shape
top_k_distances[0]

# hago una prediccion ->
preds = _numpy.zeros(ratings_train.T.shape)

for i in range(ratings_train.T.shape[0]):
    if(i%50==0):
        print("iter "+str(i))
    den = 1
    if (_numpy.abs(top_k_distances[i]).sum(axis=0)>0):
        den = _numpy.abs(top_k_distances[i]).sum(axis=0)
    preds[i, :] = top_k_distances[i].dot(ratings_train.T[top_k_items][i])/_numpy.array([den]).T

# obtengo los errores cuadráticos de las predicciones
# para el conjunto de entrenamiento y el de prueba ->
get_mse(preds, ratings_train)

get_mse(preds, ratings_test)

# FIN Filtro colaborativo basado en los KNN


