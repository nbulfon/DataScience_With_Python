#imports

from scipy.spatial import distance_matrix
import pandas as _pandas

filepath = r"C:\\Nicolas\\Curso MachineLearning with Python\\datasets\\movies"
file = filepath + "\\" + "movies.csv"
data = _pandas.read_csv(file, sep=";")
data

# saco la columna user_id
movies = data.columns.values.tolist()[1:]
movies

# (en este caso no hace falta normalizar pq se presupone que todas las peliculas vienen valoradas del 0 al 10)

# armo la matriz de distancia ->

# distancia de Manhatan:
dd1 = distance_matrix(data[movies], data[movies],p=1)    
    
# distancia Euclídea:
dd2 = distance_matrix(data[movies], data[movies], p=2)

# distancia 10
dd10 = distance_matrix(data[movies], data[movies], p=10)


# funcion para pasar a data frame una matriz de distancia
def dm_to_df(dd,col_names):
    import pandas as _pandas
    return _pandas.DataFrame(dd, index=col_names, columns=col_names)

dm_to_df(dd1,data["user_id"])
dm_to_df(dd2,data["user_id"])
dm_to_df(dd10,data["user_id"])


# representacion grafica
import matplotlib.pyplot as _plt
from mpl_toolkits.mplot3d import Axes3D

fig = _plt.figure()

ax = fig.add_subplot(111,projection="3d")
ax.scatter(xs = data["star_wars"], ys=data["lord_of_the_rings"],zs=data["harry_potter"])


## Enlaces

df = dm_to_df(dd1, data["user_id"])
df

Z = []

# uno los dos elementos mas cercanos (1,1 con 1,10) en la columna nueva, "11"
df[11] = df[1] + df[10]
df.loc[11] = df.loc[1] + df.loc[10]

# uno los puntos. 1 = punto 1. 10 = punto 10. 0.7 = distancia entre ellos. 2 = cantidad de puntos.
Z.append([1,10,0.7,2]) # id1, id2, distancia, nro de elementos en el custer.
df

# saco valores
for i in df.columns.values.tolist():
    df.loc[11][i] = min(df.loc[1][i], df.loc[10][i])
    df.loc[i][11] = min(df.loc[i][1], df.loc[i][10])
df

df = df.drop([1,10])
df = df.drop([1,10],axis=1)
df

x = 2
y = 7
n = 12

df[n] = df[x] + df[y]
df.loc[n] = df.loc[x] + df.loc[y]
Z.append([x,y, df.loc[x][y],2]) # id1, id2, distancia, nro de elementos en el custer.
for i in df.columns.values.tolist():
    df.loc[n][i] = min(df.loc[x][i], df.loc[y][i])
    df.loc[i][n] = min(df.loc[i][x], df.loc[i][y])
df

df = df.drop([x,y])
df = df.drop([x,y],axis=1)
df

x = 5
y = 8
n = 13

df[n] = df[x] + df[y]
df.loc[n] = df.loc[x] + df.loc[y]
Z.append([x,y, df.loc[x][y],2]) # id1, id2, distancia, nro de elementos en el custer.
for i in df.columns.values.tolist():
    df.loc[n][i] = min(df.loc[x][i], df.loc[y][i])
    df.loc[i][n] = min(df.loc[i][x], df.loc[i][y])
df

df = df.drop([x,y])
df = df.drop([x,y],axis=1)
df
Z

x = 11
y = 13
n = 14

df[n] = df[x] + df[y]
df.loc[n] = df.loc[x] + df.loc[y]
Z.append([x,y, df.loc[x][y],2]) # id1, id2, distancia, nro de elementos en el custer.
for i in df.columns.values.tolist():
    df.loc[n][i] = min(df.loc[x][i], df.loc[y][i])
    df.loc[i][n] = min(df.loc[i][x], df.loc[i][y])
df

df = df.drop([x,y])
df = df.drop([x,y],axis=1)
df
Z

x = 9
y = 12
z = 14
n = 15

df[n] = df[x] + df[y]
df.loc[n] = df.loc[x] + df.loc[y]
Z.append([x,y, df.loc[x][y],3]) # id1, id2, distancia, nro de elementos en el custer.
for i in df.columns.values.tolist():
    df.loc[n][i] = min(df.loc[x][i], df.loc[y][i], df.loc[z][i])
    df.loc[i][n] = min(df.loc[i][x], df.loc[i][y], df.loc[i][z])
df

df = df.drop([x,y,z])
df = df.drop([x,y,z],axis=1)
df
Z


x = 4
y = 6
z = 15
n = 16

df[n] = df[x] + df[y]
df.loc[n] = df.loc[x] + df.loc[y]
Z.append([x,y, df.loc[x][y],3]) # id1, id2, distancia, nro de elementos en el custer.
for i in df.columns.values.tolist():
    df.loc[n][i] = min(df.loc[x][i], df.loc[y][i], df.loc[z][i])
    df.loc[i][n] = min(df.loc[i][x], df.loc[i][y], df.loc[i][z])
df

df = df.drop([x,y,z])
df = df.drop([x,y,z],axis=1)
df
Z

x = 3
y = 16
n = 17

df[n] = df[x] + df[y]
df.loc[n] = df.loc[x] + df.loc[y]
Z.append([x,y, df.loc[x][y],2]) # id1, id2, distancia, nro de elementos en el custer.
for i in df.columns.values.tolist():
    df.loc[n][i] = min(df.loc[x][i], df.loc[y][i])
    df.loc[i][n] = min(df.loc[i][x], df.loc[i][y])
df

df = df.drop([x,y])
df = df.drop([x,y],axis=1)
df
Z


## FIN Enlaces

## Clustering Jerarquico
import matplotlib.pyplot as _plt
from  scipy.cluster.hierarchy import dendrogram, linkage

Z = linkage(data[movies],"ward")
Z
    
_plt.figure(figsize=(25,10))
_plt.title("Dendrogrma jerárquico para el Clustering")
_plt.xlabel("ID de los usuarios de Netflix")
_plt.ylabel("Distancia")
dendrogram(Z, leaf_rotation=90.,leaf_font_size=10.0)
_plt.show()

Z = linkage(data[movies],"average")
Z
    
_plt.figure(figsize=(25,10))
_plt.title("Dendrogrma jerárquico para el Clustering")
_plt.xlabel("ID de los usuarios de Netflix")
_plt.ylabel("Distancia")
dendrogram(Z, leaf_rotation=90.,leaf_font_size=10.0)
_plt.show()

Z = linkage(data[movies],"complete")
Z
    
_plt.figure(figsize=(25,10))
_plt.title("Dendrogrma jerárquico para el Clustering")
_plt.xlabel("ID de los usuarios de Netflix")
_plt.ylabel("Distancia")
dendrogram(Z, leaf_rotation=90.,leaf_font_size=10.0)
_plt.show()


Z = linkage(data[movies],"single")
Z
    
_plt.figure(figsize=(25,10))
_plt.title("Dendrogrma jerárquico para el Clustering")
_plt.xlabel("ID de los usuarios de Netflix")
_plt.ylabel("Distancia")
dendrogram(Z, leaf_rotation=90.,leaf_font_size=10.0)
_plt.show()


Z = linkage(data[movies],"weighted")
Z
    
_plt.figure(figsize=(25,10))
_plt.title("Dendrogrma jerárquico para el Clustering")
_plt.xlabel("ID de los usuarios de Netflix")
_plt.ylabel("Distancia")
dendrogram(Z, leaf_rotation=90.,leaf_font_size=10.0)
_plt.show()
## FIN Clustering Jerarquico
















