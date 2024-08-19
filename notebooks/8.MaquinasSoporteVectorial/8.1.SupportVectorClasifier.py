# Linear Support Vector Clasifier (ALGORITMO SUPERVISADO)

# Objetivo de un Support Vector Clasifier: intentar encajar los datos que le proporcionemos,
# devolviendonos, el mejor hiperplano que sea capaz de dividirlos, separarlos y categorizarlos.
# Luego, a partir de eso, uno puede ir agregando elementos, para saber qué clase predecida
# nos ha llevado acá.

import numpy as _numpy
import matplotlib.pyplot as _plt
from matplotlib import style
style.use("ggplot")
from sklearn import svm

X = [1,5,1.5,8,1,9]
Y = [2,8,1.8,8,0.6,11]
_plt.scatter(X,Y)
_plt.show()

#--
data = _numpy.array(list(zip(X,Y)))
data

# etiquetado de los datos (0 y 1).
target = [0,1,0,1,0,1]

# creo mi clasificador (FASE DE APRENDIZAJE) ->
# kernel: es el tipo de clasificador. Si = 'linear' quiere decir que separa una recta.
# C: es una forma de evaluar 'como de mal' queremos que sea la clasificación.
classifier = svm.SVC(kernel="linear", C=1.0)
classifier.fit(data,target)

# veo la separacion (FASE DE PREDICCIÓN) ->
# hago que el classifier prediga donde caerán los puntos que le paso.
punto = _numpy.array([10.32,12.67]).reshape(1,2)
print(punto)
classifier.predict(punto)

# establezco los clasificadores
# a = pendiente de la recta.
# b = ordenada al origen
# Modelo w0. x + w1 * y + e = 0
# y = a*X +b  -> Ecuación del Hiperplano en 2d. (Puede extenderse a n-dimensiones).
w = classifier.coef_[0]
w

a = - w[0]/w[1]
a

b = -classifier.intercept_[0] / w[1]
b

xx = _numpy.linspace(0, 10)
yy = a * xx +b
_plt.plot(xx,yy, 'k-', label="Hiperplano de separación")
_plt.scatter(X, Y, c=target)
_plt.plot()








