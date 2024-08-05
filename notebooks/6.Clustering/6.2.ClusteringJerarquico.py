# imports

import matplotlib.pyplot as _plt
from  scipy.cluster.hierarchy import dendrogram, linkage, cophenet
import numpy as _numpy
from scipy.spatial.distance import pdist


# X datasets (array de n x m) de puntos a clusterizar.
# n número de datos
# m número de rasgos
# Z array de enlace del Cluster con la información de las uniones
# k número de Clusters.

# primero establezco una semilla para la posterior generación aleatoria ->
_numpy.random.seed(4711) # elijo el nro 4711 pq se me cantó...
a = _numpy.random.multivariate_normal([10,0],[[3,1],[1,4]], size=[100,])
a
b = _numpy.random.multivariate_normal([0,20],[[3,1],[1,4]], size = 50)
b
X = _numpy.concatenate((a,b))
print(X.shape)
_plt.scatter(X[:,0], X[:,1])
_plt.show()

Z = linkage(X, "ward")
Z

# compruebo la precisión con respecto a los clustes generados.
x, coph_dist = cophenet(Z, pdist(X))

Z[152-len(X)] # Cluster 152

# veo graficamente los puntos rojos que quedaron aglutinados dada su cercania ->
idx = [33,62,68]
_plt.figure(figsize=(10,8))
_plt.scatter(X[:,0], X[:,1])
_plt.scatter(X[idx,0], X[idx,1], c='r')

## Representación gráfica de un Dendrograma

_plt.figure(figsize=(25,10))
_plt.title("Dendrogrma del Clustering jerárquico")
_plt.xlabel("Indices de la muestra")
_plt.ylabel("Distancia")
dendrogram(Z, leaf_rotation=90.,leaf_font_size=8.0, color_threshold=0.1*180)
_plt.show()
## FIN Representación gráfica de un Dendrograma


## Truncar el Dendrograma (se contraen los elemenos)

_plt.figure(figsize=(25,10))
_plt.title("Dendrogrma del Clustering jerárquico truncado")
_plt.xlabel("Indices de la muestra")
_plt.ylabel("Distancia")
dendrogram(Z, leaf_rotation=90.,leaf_font_size=8.0, color_threshold=0.1*180,
           truncate_mode="lastp",p=10,show_leaf_counts=True, show_contracted=True)
_plt.show()

## FIN Truncar el Dendrograma

## Dendrograma Tuneado
def dendrogram_tune(*args, **kwargs):
    
    max_d=kwargs.pop("max_d", None)
    if max_d and 'color_threshold' not in kwargs:
        kwargs['color_threshold'] = max_d
    annotate_above = kwargs.pop('annotate_above', 0)
    
    ddata = dendrogram(*args,**kwargs)
        
    if not kwargs.get('no_plot', False):
        _plt.title("Clustering jerárquico con Dendrograma truncado")
        _plt.xlabel("Índice del Dataset (o tamaño del cluster)")
        _plt.ylabel("Distancia")
        for i, d, c in zip(ddata['icoord'], ddata['dcoord'], ddata['color_list']):
            x = 0.5 * sum(i[1:3])
            y = d[1]
            if y>annotate_above:
                _plt.plot(x,y,'o',c=c)
                _plt.annotate('%.3g'%y, (x,y), xytext=(0,-5),
                            textcoords="offset points", va="top", ha="center")
                
    if max_d:
        _plt.axhline(y=max_d, c='k')
        
    return ddata

dendrogram_tune(Z,truncate_mode='lastp',p=12, leaf_rotation=90., leaf_font_size=12.,
                show_contracted=True,annotate_above=10, max_d=20)
_plt.show()
## FIN Dendrograma Tuneado

## Corte automático del Dendrograma

# formula del param de inconsistencia:
    # inconsistency_i = (h_i-avg(h_j)) / std(h_j)

from scipy.cluster.hierarchy import inconsistent

# elijo la profundidad (los factores de inconsistencia son super dependientes de la profundidad del arbol...)
depth = 3
incons = inconsistent(Z,depth)
incons[-10:]
#_plt.hist(incons)

# Metodo del Codo

last = Z[-10:,2]
last_rev = last[::-1]
idx = _numpy.arange(1,len(last)+1)
_plt.plot(idx, last_rev)


acc = _numpy.diff(last,2)
acc_rev = acc[::-1]
_plt.plot(idx[:-2]+1, acc_rev)
_plt.show()
k = acc_rev.argmax()+2
print("El número optimo de Clusters es "+str(k))
# FIN Metodo del Codo

c = _numpy.random.multivariate_normal([40,40],[[20,1],[1,30]], size=[200,])
d = _numpy.random.multivariate_normal([80,80],[[30,1],[1,30]], size=[200,])
e = _numpy.random.multivariate_normal([0,100],[[100,1],[1,100]], size=[200,])
X2 = _numpy.concatenate((X,c,d,e),)
_plt.scatter(X2[:,0], X2[:,1])
_plt.show()

Z2 = linkage(X2,"ward")
_plt.figure(figsize=(10,10))
dendrogram_tune(
    Z2,
    truncate_mode="lastp",
    p=30,
    leaf_rotation=90.,
    leaf_font_size=10.,
    show_contracted=True,
    annotate_above = 40,
    max_d = 170
)

_plt.show()

last = Z2[-10:,2]
last_rev = last[::-1]
print(last_rev)
idx = _numpy.arange(1, len(last)+1)
_plt.plot(idx, last_rev)

acc = _numpy.diff(last,2)
acc_rev = acc[::-1]
_plt.plot(idx[:-2]+1, acc_rev)
_plt.show()
k = acc_rev.argmax() +2
print("El número óptimo de cluster es %s"%str(k))

print(inconsistent(Z2, 5)[-10:])
## FIN Corte automático del Dendrograma



# Recuperar los clusters y sus elementos

from scipy.cluster.hierarchy import fcluster

# seteo la distancia de corte para los clusters ->
max_d=25
clusters = fcluster(Z, max_d, criterion="distance") # 1.matriz de enlance, 2.distancia de corte, 3.criterio de decision.
clusters

k = 3
clusters = fcluster(Z, k, criterion="maxclust")
clusters

fcluster(Z, 6, depth=10)

_plt.figure(figsize=(10,8))
_plt.scatter(X[:,0], X[:,1], c = clusters, cmap="prism")
_plt.show()

max_d=170
clusters = fcluster(Z2, max_d, criterion="distance")
clusters

_plt.figure(figsize=(10,8))
_plt.scatter(X2[:,0], X2[:,1], c = clusters, cmap="prism")
_plt.show()
# FIN Recuperar los clusters y sus elementos




















