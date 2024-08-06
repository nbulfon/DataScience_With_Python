from math import sin, cos, radians, pi, sqrt
import numpy.random as rnd
import numpy as np
import matplotlib.pyplot as plt

def ring(r_min = 0, r_max = 1, n_samples = 360):
    angle = rnd.uniform(0, 2*pi, n_samples)
    distance = rnd.uniform(r_min, r_max, n_samples)
    data = []
    for a, d in zip(angle, distance):
        data.append([d*cos(a), d*sin(a)])
    return np.array(data)

data1 = ring(3, 5)
data2 = ring(24, 27)

data = np.concatenate([data1, data2], axis = 0)
labels = np.concatenate([[0 for i in range(0,len(data1))], [1 for i in range(0,len(data2))]])
plt.scatter(data[:,0], data[:,1], c = labels, s = 5, cmap = "autumn")


#Algoritmo con Kmeans

from sklearn.cluster import KMeans

km = KMeans(2).fit(data)

clust = km.predict(data)

plt.scatter(data[:,0], data[:, 1], c = clust, s = 5, cmap="autumn")


#Algoritmo de los K Medoides

from pyclust import KMedoids

kmed = KMedoids(2).fit_predict(data)

plt.scatter(data[:,0], data[:,1], c=kmed, s=5, cmap="autumn")


#Algoritmo del Clustering Espectral

from sklearn.cluster import SpectralClustering

clust = SpectralClustering(2).fit_predict(data)

plt.scatter(data[:,0], data[:,1], c = clust, s = 5, cmap = "autumn")


#Podemos estimar la k:

#    No: Propagación de la afinidad
#    Si: Podemos usar la distancia Euclídea:
#        Si: K-Means
#        No: Buscar valores centrales:
#            Si: K-Medoides
#            No: Los datos son linealmente separables:
#                Si: Clustering aglomerativo
#                No: Clustering Espectral




