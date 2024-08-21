# An√°lisis de Componentes Principales - Paso a Paso

#  -  Estandarizar los datos (para cada una de las m observaciones)
#  -  Obtener los vectores y valores propios a partir de la matriz de covarianzas
# o de correlaciones o incluso la t√©cnica de singular vector decomposition.
#  -  Ordenar los valores propios en orden descendente y quedarnos con los p que se correpondan
# a los p mayores y as√≠ disminuir el n√∫mero de variables del dataset (p<m)
#  -  Constrir la matriz de proyecci√≥n W a partir de los p vectores propios
#  -  Transformar el dataset original X a trav√©s de W (matr√≠z 'A' en la demostraciÛn matem√°tica)
# para as√≠ obtener datos en el subespacio dimensional de dimensi√≥n p, que ser√° Y.

# imports
import pandas as _pandas

df = _pandas.read_csv(r"C:\NICOLAS\Curso MachineLearning with Python\datasets\iris\iris.csv")

# separo var. explicativas y explicada (reduzo dimensionalidad)->
X = df.iloc[:,0:4].values
y = df.iloc[:,4].values

import chart_studio.plotly as _plotly
import plotly.graph_objects as go
import plotly.offline as pyo


# Configura las credenciales solo si est·s subiendo gr·ficos a Chart Studio
import chart_studio
chart_studio.tools.set_credentials_file(
    username='nbulfon', api_key='sEFrT9ng2wtZgch44nOM')

traces = []
legend = {0: True, 1: True, 2: True, 3: True}

colors = {'setosa': 'rgb(255,127,20)',
          'versicolor': 'rgb(31, 220, 120)',
          'virginica': 'rgb(44, 50, 180)'}

# Suponiendo que X y y est·n definidos
for col in range(4): 
    for key in colors:
        traces.append(go.Histogram(x=X[y == key, col],
                                   opacity=0.7, 
                                   xaxis="x%s" % (col + 1),
                                   marker={"color": colors[key]},
                                   name=key, showlegend=legend[col])
                     )
        
    legend = {0: False, 1: False, 2: False, 3: False}

layout = go.Layout(
    title={"text": "DistribuciÛn de los rasgos de las diferentes flores Iris",
           "xref": "paper", "x": 0.5},
    barmode="overlay",
    xaxis={"domain": [0, 0.25], "title": "Long. SÈpalos (cm)"},
    xaxis2={"domain": [0.3, 0.5], "title": "Anch. SÈpalos (cm)"},
    xaxis3={"domain": [0.55, 0.75], "title": "Long. PÈtalos (cm)"},
    xaxis4={"domain": [0.8, 1.0], "title": "Anch. PÈtalos (cm)"},
    yaxis={"title": "N˙mero de ejemplares"}
)

fig = go.Figure(data=traces, layout=layout)

# Mostrar el gr·fico
fig.show()
# muestro en un browser ->
pyo.plot(fig)

from sklearn.preprocessing import StandardScaler

X_std = StandardScaler().fit_transform(X)

traces = []
legend = {0:True, 1:True, 2:True, 3:True}

colors = {'setosa': 'rgb(255,127,20)',
         'versicolor': 'rgb(31, 220, 120)',
         'virginica': 'rgb(44, 50, 180)'}


for col in range(4): 
    for key in colors:
        traces.append(go.Histogram(x=X_std[y==key, col],
                                   opacity = 0.7, 
                                   xaxis="x%s"%(col+1),
                                   marker={"color":colors[key]},
                                   name = key, showlegend=legend[col])
                     )
        
    legend = {0:False, 1:False, 2:False, 3:False}

layout = go.Layout(
    title={"text":"DistribuciÛn de los rasgos de las diferentes flores Iris",
           "xref" : "paper","x" : 0.5},
    barmode="overlay",
    xaxis= {"domain" : [0,0.25], "title":"Long. SÈpalos (cm)"},
    xaxis2= {"domain" : [0.3, 0.5], "title" : "Anch. SÈpalos (cm)"},
    xaxis3= {"domain" : [0.55, 0.75], "title" : "Long. PÈtalos (cm)"},
    xaxis4= {"domain" : [0.8,1.0], "title" : "Anch. PÈtalos (cm)"},
    yaxis={"title":"N˙mero de ejemplares"}
)

fig = go.Figure(data=traces, layout=layout)
_plotly.iplot(fig)
# muestro en un browser ->
pyo.plot(fig)
#fig.show()


# Calculamos la descomposiciÛn de valores y vectores propios.
## a) Usando la Matriz de Covarianzas
from IPython.display import display, Math, Latex

display(Math(r'\sigma_{jk} = \frac{1}{n-1}\sum_{i=1}^m (x_{ij} - \overline{x_j})(x_{ik} - \overline{x_k})'))

display(Math(r'\Sigma = \frac{1}{n-1}((X-\overline{x})^T(X-\overline{x}))'))

display(Math(r'\overline{x} = \sum_{i=1}^n x_i\in \mathbb R^m'))
import numpy as _numpy

mean_vect = _numpy.mean(X_std, axis=0)
mean_vect

# la matriz de covarianza debe tener necesariamente por la demostraciÛn previa,
# su diagonal principal con el valor '1'. Viene por los problemas arrastrados...
cov_matrix = (X_std - mean_vect).T.dot((X_std - mean_vect))/(X_std.shape[0]-1)
print("La matriz de covarianzas es \n%s"%cov_matrix)

# hago lo mismo, con numpy-->
_numpy.cov(X_std.T)

# descomposiciÛn de la matriz de convarianzas ->
eig_vals, eig_vectors = _numpy.linalg.eig(cov_matrix)
print("Autovalores \n%s"%eig_vals)
print("Autovectores \n%s"%eig_vals)

## b) Usando la Matriz de Correlaciones (muy ˙til en el campo de las finanzas).
# (se normalizan los elementos de la diagonal principal. Es decir, son int y no float...)
corr_matrix = _numpy.corrcoef(X_std.T)
corr_matrix

eig_vals_corr, eig_vectors_corr = _numpy.linalg.eig(corr_matrix)
print("Autovalores \n%s"%eig_vals_corr)
print("Autovectores \n%s"%eig_vectors_corr)

# calculo la matriz de ocrrelaciones del datos in estandarizar
corr_matrix = _numpy.corrcoef(X.T)
corr_matrix

## c) Singular Value Decomposition (la m·s usada por el MachineL por su eficacia computacional).
u,s,v = _numpy.linalg.svd(X_std.T)
u
s
v


# 2 - Las componentes principales
for ev in eig_vectors:
    print("La longitud del autovector es: ",_numpy.linalg.norm(ev)) # deben ser = 1.

# junto los autovalores y autocectores en parejas ->
eigen_pairs = [(_numpy.abs(eig_vals[i]), eig_vectors[:,i]) for i in range(len(eig_vals))]
eigen_pairs

# Ordenamos los autovectores con autovalor de mayor a menor
eigen_pairs.sort()
eigen_pairs.reverse()
eigen_pairs

print("Autovalores en orden descendente:")
for ep in eigen_pairs:
    print(ep[0]) # recordar que c/u de los autovalores, es la varianza de la nueva variable.

# ø con cuantos valores me quedo ? --> uso la 'Varianza Explicativa'.
total_sum = sum(eig_vals)
var_exp = [(i/total_sum)*100 for i in sorted(eig_vals, reverse=True)]
cum_var_exp = _numpy.cumsum(var_exp)

# hago gr·ficamente, la VARIANZA EXPLICADA ACUMULADA ->
plot1 = go.Bar(x=[f"CP {i}" for i in range(1,5)], y=var_exp, showlegend= True)
plot2 = go.Scatter(x=[f"CP {i}" for i in range(1,5)], y=cum_var_exp, showlegend= True)

data = [plot1,plot2]

layout = go.Layout(xaxis= {"title": "Componentes principales"},
                  yaxis ={"title": "Porcentaje de varianza explicada"},
                  title = "Porcentaje de variabilidad explicada por cada componente principal")

fig = go.Figure(data=data,layout=layout)
_plotly.iplot(fig)
pyo.plot(fig)
# PARA NOTAR ESTO, VER QUE LAS COLUMNAS MAS CHICAS (VARIANZA ACUMULADA MENOR)
# SON AQUELLAS LAS CUALES EXPLICAN MENOR O POCA CANTIDAD PARA EL MODELO, ES DECIR,
# PODRIAMOS ELIMINARLAS Y QUEDARNOS CON LAS COLUMNAS (VARS) QUE MAS EXPLIQUEN NUESTRO MODELO
# A FIN DE REDUCIR LA DIMENSIONALIDAD Y SER MAS EFICIENTES COMPUTACIONALMENTE.

#fig.show()

# la matriz W = la matriz A en la demostraciÛn matem·tica.
W = _numpy.hstack((eigen_pairs[0][1].reshape(4,1), 
               eigen_pairs[1][1].reshape(4,1)))
W

# 3- Proyectando las variables en el nuevo subespacio vectorial
# con esto voy a poder trabajar con menos dimensiones de las cuales partÌ inicialmente.
Y = X_std.dot(W)
result = []

results = []
for name in ('setosa', 'versicolor', 'virginica'):
    result = go.Scatter(x= Y[y==name,0], y =Y[y==name, 1],
                       mode = "markers", name=name,
    marker= { "size": 12, "line" : { "color" : 'rgba(220,220,220,0.15)', "width":0.5},
           "opacity": 0.8})
    results.append(result)
    
layout = go.Layout(showlegend = True, 
                   scene ={ "xaxis" :{"title": "Componente Principal 1"},
                            "yaxis" : {"title": "Componente Principal 2"}},
                  xaxis ={ "zerolinecolor": "gray"},
                  yaxis={ "zerolinecolor": "gray"})
fig = go.Figure(data=results,layout=layout)
_plotly.iplot(fig)
pyo.plot(fig)
#fig.show()




















