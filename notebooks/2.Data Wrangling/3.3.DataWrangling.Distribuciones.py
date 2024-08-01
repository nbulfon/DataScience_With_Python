# Distribuciones de probabilidad

import numpy as _numpy
import matplotlib.pyplot as _matplotlib


# Distribución UNIFORME
# a = limite inferior. b = limite superior
# unica restriccion (logica): a < b
a = 1
b = 100
n = 200000000
data = _numpy.random.uniform(a,b,n)

# grafico
_matplotlib.hist(data)
# FIN Distribución UNIFORME


# Distribución NORMAL
data2 = _numpy.random.randn(100)
x = range(1,101)
_matplotlib.plot(x,data2)

_matplotlib.hist(data2,color="green")

# ordeno los datos ascendentemente con sorted para notar la distribucion Gaussiana ->
_matplotlib.plot(x, sorted(data2))

media_mu = 5.5
standard_desviation = 2.5
# despejando media:
# Z = (X - media_mu) / standard_desviation  --> N(0,1) , X = media_mu + standard_desviation* Z
data3 = media_mu + standard_desviation* _numpy.random.randn(10000)
_matplotlib.hist(data3)
# FIN Distribución NORMAL


















