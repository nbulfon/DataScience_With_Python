# Juntando R y Python

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import rpy2.robjects as ro
import rpy2.robjects.numpy2ri

# Activar la conversión automática entre objetos de NumPy y R
rpy2.robjects.numpy2ri.activate()

# Definir la función R en Python
codigo_r = """
saludar <- function(cadena){
    return(paste("Hola, ", cadena))
}
"""

ro.r(codigo_r)

# Obtener la función R en el entorno Python
saludar_py = ro.globalenv["saludar"]

# Usar la función R desde Python
res = saludar_py("Antonio Banderas")
print(res[0])

# Verificar el tipo del objeto retornado
print(type(res))

# Imprimir la representación R de la función
print(saludar_py.r_repr())

# Pasar un vector de Python a R
var_from_python = ro.FloatVector(np.arange(1, 5, 0.1))
print(var_from_python.r_repr())

# Asignar la variable a R
ro.globalenv["var_to_r"] = var_from_python

# Realizar operaciones en R
print(ro.r("sum(var_to_r)"))
print(ro.r("mean(var_to_r)"))
print(ro.r("sd(var_to_r)"))
print(ro.r("summary(var_to_r)"))

# Graficar el histograma en R y mostrarlo en Python
ro.r("hist(var_to_r, breaks = 4)")
plt.show()

# Trabajar de forma conjunta entre R y Python

from rpy2.robjects.packages import importr

# Instalar paquetes de R desde Python
ro.r('options(repos = "http://cran.rstudio.com/")')
ro.r("install.packages('extRemes')")  # Instalar paquete de R

# Importar el paquete extRemes de R
extremes = importr("extRemes")

# Usar la función fevd del paquete extRemes
fevd = extremes.fevd

# Cargar datos en Python
data = pd.read_csv("../datasets/time/time_series.txt", 
                   sep=r"\s*", skiprows=1, parse_dates=[[0, 1]],
                   names=["date", "time", "wind_speed"],
                   index_col=0)

# Mostrar los primeros datos
print(data.head(5))

# Agrupar por año y obtener la velocidad máxima del viento
max_ws = data.wind_speed.groupby(pd.Grouper(freq="A")).max()

# Graficar las velocidades máximas
max_ws.plot(kind="bar", figsize=(16, 9))
plt.show()

# Realizar ajuste GEV usando R desde Python
result = fevd(max_ws.values, type="GEV", method="GMLE")
print(type(result))
print(result.r_repr())

# Extraer resultados desde el objeto R
res = result.rx("results")[0]
loc, scale, shape = res.rx("par")[0]

print("Loc:", loc[0])
print("Scale:", scale[0])
print("Shape:", shape[0])

# Realizar ajustes adicionales y graficar
metodos = ["MLE", "GMLE", "Bayesian", "Lmoments"]
tipos = ["GEV", "Gumbel"]

for t in tipos:
    for m in metodos:
        print("Tipo de Ajuste:", t)
        print("Método del Ajuste:", m)
        result = fevd(max_ws.values, method=m, type=t)
        print(result.rx("results")[0])
        
        # Graficar el resultado del ajuste
        ro.globalenv['result'] = result
        ro.r('plot.fevd(result)')
        plt.show()
