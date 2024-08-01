# imports
import pandas as _pandas
import numpy as _numpy
import matplotlib.pyplot as _plt

## Modelo con datos simulados:
    # y = a + b*X
    #=> X: 100 valores distribuidos segun una N(1.5 , 2.5)
    #Ye = 5 + 1.9 * X + e # los valores son arbitrarios
    # e estará distrobuido segun una N(0 , 0.8)
    
x = 1.5 + 2.5 * _numpy.random.randn(100)
residuos = 0 + 0.8 * _numpy.random.randn(100)

y_pred = 5+ 1.9*x
y_act = 5 + 1.9 *x + residuos

x_list = x.tolist()
y_pred_list = y_pred.tolist()
y_act_list = y_act.tolist()

data = _pandas.DataFrame({
        "x":x_list,
        "y_actual":y_act_list,
        "y_prediccion":y_pred
    })

y_mean = [_numpy.mean(y_act) for i in range(1, len(x_list) +1)]

_plt.plot(x,y_pred)
_plt.plot(x,y_act,"ro")
_plt.plot(x,y_mean)
_plt.title("Valor Actual Vs Predicción")


# --
# la Suma de los cuadrados de la regresion mide la diferencia entre las y estimadas y la recta de regresion
data["SSR"] = (data["y_prediccion"]-_numpy.mean(y_act)) **2

data["SSD"] = (data["y_prediccion"] - data["y_actual"])**2
data["SST"] = (data["y_actual"] - _numpy.mean(y_act))**2


data.head()
SSR = sum(data["SSR"])
SSD = sum(data["SSD"])
SST = sum(data["SST"])

R2 = SSR/SST

_plt.plot(data["SSD"])
_plt.hist(data["y_prediccion"] - data["y_actual"])

## Obteniendo la recta de regresión

# METODO MCO

# y = a + b * x
# b = sum((x_iesimo - x_media) * (y_iesimo - y _media) / sum(x_iesimo - x_media))
# a = y_media - b * x_media

x_mean = _numpy.mean(data["x"])
y_mean = _numpy.mean(data["y_actual"])

data["covarianza_xy"] = (data["x"] - x_mean) * (data["y_actual"] - y_mean)
data["varianza_x"] = (data["x"] - x_mean) **2

beta = sum(data["covarianza_xy"]) / sum(data["varianza_x"])
alpha = y_mean - beta * x_mean
print(beta)
print(alpha)



data["y_model"] = alpha + beta * data["x"]
data.head()

SSR = sum((data["y_model"]-_numpy.mean(y_mean)) **2)
SSD = sum((data["y_model"] - data["y_actual"])**2)
SST = sum((data["y_actual"] - y_mean)**2)

R2 = SSR / SST

y_mean = [_numpy.mean(y_act) for i in range(1, len(x_list) +1)]
_plt.plot(data["x"],data["y_prediccion"])
_plt.plot(data["x"], data["y_actual"],"ro")
_plt.plot(data["x"],y_mean, "g")
_plt.plot(data["x"], data["y_model"])
_plt.title("Valor Actual Vs Predicción")


# P-valor --> para significancia individual.
# Si el p-valor resultante es menor que el nivel de significación, rechazamos la 
# hipótesis nula y aceptamos que existe una relación lineal entre x e y.

# Estadistico F ---> para la significatividad conjunta.
# Este estadistico se usa cuando se trabaja con una regresion lineal multiple
# con muchas variables...porque puede pasar que si hay muchas variables,
# algunas tengan relacion entre si y se influyan mutuamente.

# RSE --> Error Estándar Residual. Es inevitable.
# RSE es la desviación estándar del término del error (desviación de la parte
# de datos que el modelo n oes capaz de explicar por falta de información o más datos
# adicionales )
# Generalmente, RSE debería ir decreciendo a medida que tenemos más variables.
# RSE - calculo
# RSE = raizCuadrada(SSD / n- k - 1)
RSE = _numpy.sqrt(SSD / len(data) -2)
print(RSE)
RSE / _numpy.mean(data["y_actual"])














