# imports
import numpy as _numpy

data = _numpy.random.random(90).reshape(30,3) # el reshape los organiza segun las filas y columnas que le pase.
data

# elijo dos centroides al azar para implementar luego por el metodo K-means
centroide_1 =  _numpy.random.choice(range(len(data)))
centroide_2 = _numpy.random.choice(range(len(data)))
clust_centers = _numpy.vstack( [data[centroide_1],data[centroide_2]] )
clust_centers

# Metodo K-Means
from scipy.cluster.vq import vq, kmeans

# muestro a qué cluster pertenece cada observacion ->
# el segundo array da la distancia de c/u de las 3o observaciones, del centro del clustering final
vq(data,clust_centers)

# obtengo la info. de los varicentros. En donde está el centro de cada Cluster ->
kmeans(data, clust_centers)




