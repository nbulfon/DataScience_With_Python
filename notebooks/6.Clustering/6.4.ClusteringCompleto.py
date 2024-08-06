# Imports

import numpy as _numpy
import pandas as _pandas
import matplotlib.pyplot as _plt

## Etapa 1: Importar el dataset

#filepath = r"C:\\Nicolas\\Curso MachineLearning with Python\\datasets\\wine"
filepath  = r"C:\\Nicolas\\DataScience_With_Python\\datasets\\wine"
file = filepath + "\\" + "winequality-red.csv"
df = _pandas.read_csv(file, sep=";")
df

_plt.hist(df["quality"])

df.groupby("quality").mean()

## Etapa 2: Normalización de los datos --defino el rango de valores en [0,1]--

df_normalizado = (df-df.min())/(df.max()-df.min())
df_normalizado

## Etapa 3 Clustering jerárquico con scikit-learn
from sklearn.cluster import AgglomerativeClustering

clus = AgglomerativeClustering(n_clusters=6, linkage="ward").fit(df_normalizado)
md_Herarchy = _pandas.Series(clus.labels_)

_plt.hist(md_Herarchy)
_plt.title("Histograma de los clusters")
_plt.xlabel("Cluster")
_plt.ylabel("Número de vinos del cluster")

clus.children_

from scipy.cluster.hierarchy import dendrogram, linkage

# a la matriz de enlace Z, le paso el dataset normalizado y el método de ward para hacer el enlace.
Z = linkage(df_normalizado,"ward")
_plt.figure(figsize=(25,10))
_plt.title("Dendrograma de los vinos")
_plt.xlabel("ID del vino")
_plt.ylabel("Distancia")
_plt.show()
dendrogram(Z, leaf_rotation=90., leaf_font_size=8.)


## Clustering con K-means
from sklearn.cluster import KMeans
from sklearn import datasets

model = KMeans(n_clusters=6)
model.fit(df_normalizado)

md_kMeans = _pandas.Series(model.labels_)

df_normalizado["clust_herarchy"] = md_Herarchy
df_normalizado["clust_kMeans"] = md_kMeans
df_normalizado
_plt.hist(md_kMeans)


## Etapa 4 Interpretación final
model.cluster_centers_

df_normalizado.groupby("clust_kMeans").mean()
























