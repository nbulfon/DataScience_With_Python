# Juntando R y Python

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import rpy2.robjects as ro
import rpy2.robjects.numpy2ri

rpy2.robjects.numpy2ri.activate()

codigo_r = """
saludar <- function(cadena){
    return(paste("Hola, ", cadena))
}
"""

ro.r(codigo_r)

saludar_py = ro.globalenv["saludar"]

res = saludar_py("Antonio Banderas")
res[0]

type(res)

print(saludar_py.r_repr())

var_from_python = ro.FloatVector(np.arange(1,5,0.1))

var_from_python

print(var_from_python.r_repr())
ro.globalenv["var_to_r"] = var_from_python

ro.r("var_to_r")
ro.r("sum(var_to_r)")
ro.r("mean(var_to_r)")
ro.r("sd(var_to_r)")
np.sum(var_from_python)
np.mean(var_from_python)
ro.r("summary(var_to_r)")
ro.r("hist(var_to_r, breaks = 4)")


# Trabajar de forma conjunta entre R y Python

from rpy2.robjects.packages import importr

ro.r("install.packages('extRemes')")# si os falla decidle 'n' al hacer la instalación

extremes = importr("extRemes") # library(extRemes)

fevd = extremes.fevd

print(fevd.__doc__)

data = pd.read_csv("../datasets/time/time_series.txt", 
                   sep = "\s*", skiprows = 1, parse_dates = [[0,1]],
                   names = ["date", "time", "wind_speed"],
                   index_col = 0)

data.head(5)
data.shape
max_ws = data.wind_speed.groupby(pd.TimeGrouper(freq="A")).max()
max_ws.plot(kind="bar", figsize=(16,9))
result = fevd(max_ws.values, type="GEV", method = "GMLE")
print(type(result))
result.r_repr

print(result.names)


res = result.rx("results")

print(res[0])

loc, scale, shape = res[0].rx("par")[0]

loc

23.06394151991562

scale

1.7576912874286912

shape

# Función mágica para R
#%load_ext rpy2.ipython
help(rpy2.ipython.rmagic.RMagics.R)

# Un ejemplo complejo de R, Python y Rmagic

metodos = ["MLE", "GMLE", "Bayesian", "Lmoments"]
tipos = ["GEV", "Gumbel"]

for t in tipos:
    for m in metodos:
        print("Tipo de Ajuste: ", t)
        print("Método del Ajuste: ", m)
        result = fevd(max_ws.values, method = m, type = t)
        print(result.rx("results")[0])
#        %R -i result plot.fevd(result)















