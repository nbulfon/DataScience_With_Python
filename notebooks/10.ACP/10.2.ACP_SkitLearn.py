# Análisis de Componentes Principales con SkitLearn - Paso a Paso

import pandas as _pandas

import chart_studio.plotly as _plotly
import plotly.graph_objects as go
import plotly.offline as pyo
import chart_studio
from sklearn.preprocessing import StandardScaler

chart_studio.tools.set_credentials_file(
    username='nbulfon', api_key='sEFrT9ng2wtZgch44nOM')

df = _pandas.read_csv(r"C:\NICOLAS\Curso MachineLearning with Python\datasets\iris\iris.csv")


X = df.iloc[:,0:4].values
y = df.iloc[:,4].values
X_std = StandardScaler().fit_transform(X)

from sklearn.decomposition import PCA as sk_pca

acp = sk_pca(n_components=2)
Y = acp.fit_transform(X_std)

Y

results = []

for name in ('setosa', 'versicolor', 'virginica'):
    result = go.Scatter (x = Y[y==name,0], y = Y[y==name,1],
    mode = "markers", name = name, marker = {"size":8, "line": {"color": "rgba(225,225,225,0.2)","width": 0.5}}, opacity= 0.75)
    results.append(result)

layout = go.Layout(xaxis = {"title":'CP1', "showline" :False, "zerolinecolor" : "gray"}, yaxis = {"title" :'CP2', "showline" :False, "zerolinecolor" : "gray"})

fig = go.Figure(data=results, layout=layout)
_plotly.iplot(fig)
pyo.plot(fig)
#fig.show()













