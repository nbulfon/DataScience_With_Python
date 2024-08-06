# Imports

import numpy as _numpy
import pandas as _pandas
import matplotlib.pyplot as _plt

from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist
from sklearn.metrics import silhouette_samples,silhouette_score

# agarro datos iniciales a clasificar (datos inventados)
x1 = _numpy.array([3,1,1,2,1,6,6,6,5,6,7,8,9,8,9,9,8])
x2 = _numpy.array([5,4,5,6,5,8,6,7,6,7,1,2,1,2,3,2,3])
X = _numpy.array(list(zip(x1,x2))).reshape(len(x1),2)
_plt.plot()
_plt.xlim([0,10])
_plt.ylim([0,10])
_plt.title("Dataset a clasificar")
_plt.xlabel("X")
_plt.ylabel("Y")
_plt.scatter(x1, x2)
_plt.show()

max_k = 10 # máximo número de clusters que vamos a crear
K = range(1,max_k)
ssw = [] # suma de los cuadrados internos, para luego hacer el gráfico del Codo
color_palette = [_plt.cm.Spectral(float(i)/max_k) for i in K]
centroide = [sum(X)/len(X) for i in K]
sst = sum(_numpy.min(cdist(X, centroide, "euclidean"), axis = 1)) # Suma de los cuadrados totales

#  hago K-means para c/k que he definido
for k in K:
    kmeanModel = KMeans(n_clusters=k).fit(X)
    centers = _pandas.DataFrame(kmeanModel.cluster_centers_)
    labels = kmeanModel.labels_
    # mido la distancia de cada punto al baricentro respectivo
    ssw_k = sum(_numpy.min(cdist(X, kmeanModel.cluster_centers_, "euclidean"),axis=1))
    ssw.append(ssw)
    # array de colores para cada punto ->
    label_color = [color_palette[i] for i in labels]
    
    ## Fabricaremos una silueta para cada cluster (el único caso cuando no puedo es cuando k == 1 o si existen tantos clusters como puntos) ->
    # Por seguridad, no hacemos silueta si k = 1 o si k = len(X)
    if 1 < k < len(X):
        # Crear un subplot de una fila y dos columnas
        fig, (axis1,axis2) = _plt.subplots(1,2)
        fig.set_size_inches(20,8)
        # El primer subplot contendrá la silueta, que puede contener valores de [-1,1]
        # En nuestro caso, ya controlamos que los valores estén dentro de ese rango.
        axis1.set_xlim([-0.1,1.0])
        # El número de clusters a insertar determinará el tamaño de cada barra
        # El coeficient (n_clusters +1) * 10 será el espacio en blanco que dejaremos entre siluetas individuales
        # de cada cluster para separarlas.
        axis1.set_ylim([0, len(X)+ (k+1)*10])
        
        silhouette_avg = silhouette_score(X, labels)
        #print("* Para k ="+k+ "El promedio de la silueta es de:"+silhouette_avg)
        sample_silhouette_values = silhouette_samples(X, labels)
        
        y_lower = 10
        for i in range(k):
            # Agregamos la silueta del cluster k-ésimo
            ith_cluster_sv = sample_silhouette_values[labels == i]
           # print("   - Para i =",i+1,"la silueta del cluster vale : ",_numpy.mean(ith_cluster_sv))
            # ordenamos descendientemente las siluetas del cluster i-ésimo
            ith_cluster_sv.sort()
            
            ith_cluster_size = ith_cluster_sv.shape[0]
            y_upper = y_lower + ith_cluster_size
            
            # Elejimos el color del cluster
            color = color_palette[i]
            
            # Pintamos la silueta del cluster i-ésimo
            axis1.fill_betweenx(_numpy.arange(y_lower,y_upper),
                                0,ith_cluster_sv,facecolor=color,alpha=0.7)
    
            # Etiquetamos dicho cluster con el número en el centro
            axis1.text(-0.05, y_lower + 0.5 * ith_cluster_size, str(i+1))
            
            # Calculamos el nuevo y_lower para el siguiente cluster del gráfico (el siguiente tope digamos)
            y_lower = y_upper + 10 # dejamos vacías 10 posiciones sin muestra
        
        axis1.set_title("Representación de la silueta para k ="+str(k))
        axis1.set_xlabel("S(i)")
        axis1.set_ylabel("ID del Cluster")
        
        # Fin de la representación de la silueta
        
    ##Plot de los k-means con los puntos respectivos
    _plt.plot()
    _plt.xlim([0,10])
    _plt.ylim([0,10])
    _plt.title("Clustering para k = %s"%str(k))
    _plt.scatter(x1,x2, c=label_color)
    _plt.scatter(centers[0], centers[1], marker = "x")
    _plt.show()
    
    
#Representación del codo
_plt.plot(K, ssw, "bx-")
_plt.xlabel("k")
_plt.ylabel("SSw(k)")
_plt.title("La técnica del codo para encontrar el k óptimo")
_plt.show()

#Representación del codo normalizado
_plt.plot(K, 1-ssw/sst, "bx-")
_plt.xlabel("k")
_plt.ylabel("1-norm(SSw(k))")
_plt.title("La técnica del codo normalizado para encontrar el k óptimo")
_plt.show()
    



















