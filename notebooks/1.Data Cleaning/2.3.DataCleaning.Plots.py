# Plots y visualización de los datos

import pandas as _pandas
import numpy as _numpy
import matplotlib.pyplot as _matplotlib


data = _pandas.read_csv(r"C:\NICOLAS\Curso MachineLearning with Python\datasets\customer-churn-model\Customer Churn Model.txt")

print(data.head())

# Scatter plot (grafico de dispersion) ->
data.plot(kind="scatter", x="Day Mins",y="Day Charge")
data.plot(kind="scatter", x="Night Mins",y="Night Charge",color="green")

#---
figure, axs = _matplotlib.subplots(2,2,sharey=True,sharex=True)
data.plot(kind="scatter",x="Day Mins",y="Day Charge",ax=axs[0][0], color="blue")
data.plot(kind="scatter", x="Night Mins",y="Night Charge",ax=axs[0][1], color="green")
data.plot(kind="scatter", x="Day calls",y="Day Charge",ax=axs[1][0], color="yellow")
data.plot(kind="scatter", x="Night calls",y="Night Charge",ax=axs[1][1], color="red")
# FIN Scatter plot

# Histograma de frecuencias ->
k = int(_numpy.ceil(1+ _numpy.log2(3333)))
_matplotlib.hist(data["Day Calls"],bins= k) # bins = [0, 30, 60, ... 200]
_matplotlib.title("Histograma de número de llamadas al día")
_matplotlib.xlabel("Número de llamadas al día")
_matplotlib.ylabel("Frecuencias de llamadas al día")
# FIN Histogramas de frecuencias

# Diagrama de caja y bigotes
_matplotlib.boxplot(data["Day Calls"])
_matplotlib.ylabel("Número de llamadas diarías")
_matplotlib.title("Box plot de las llamadas diarías")

rango_intercuartilico = data["Day Calls"].quantile(0.75) - data["Day Calls"].quantile(0.25)

bigote_inferior = data["Day Calls"].quantile(0.25) - 1.5* rango_intercuartilico
bigote_superior = data["Day Calls"].quantile(0.25) + 1.5* rango_intercuartilico


# FIN Diagrama de caja y bigotes










