#SVM para Regresión

import numpy as _numpy
import matplotlib.pyplot as _plt

# armo los conjuntos ->
X = _numpy.sort(5*_numpy.random.rand(200,1),axis=0)
Y = _numpy.sin(X).ravel()
Y[::5] += 3*(0.5 - _numpy.random.rand(40))

_plt.scatter(X,Y, color="darkorange", label="data")

# como veo que el kernel parece importante, voy a estudiar queé efecto
# tendrá una regresión, con 3 kernels diferentes
# (lineal, radial y polinómica).

# importo Support Vector Regression
from sklearn.svm import SVR

C=1e3
# defino las 3 Support Vector Regression
svr_lin = SVR(kernel="linear", C=C)
svr_rbf = SVR(kernel="rbf", C=C, gamma=0.1)
svr_pol = SVR(kernel="poly", C=C, degree=3)

# defino los limites de y, la var. explicada.
# (lo hago asi nomás...si quisiera hacerlo bien, tomaria una muestra aleatoria,
# haria una cross-valdiation, etc...).
y_lin = svr_lin.fit(X,Y).predict(X)
y_rbf = svr_rbf.fit(X,Y).predict(X)
y_pol = svr_pol.fit(X,Y).predict(X)

lw = 2
# grafico ->
_plt.figure(figsize=(16,9))
_plt.scatter(X,Y,color="darkorange", label ="data")
_plt.plot(X,y_lin, color="green", lw = lw, label = "SVM Lineal")
_plt.plot(X,y_rbf, color="navy", lw=lw, label="SVM Radial")
_plt.plot(X,y_pol, color="cornflowerblue", label="SVM Polinómico")
_plt.xlabel("x")
_plt.ylabel("y")
_plt.title("Support Vector Regression")
_plt.legend()
_plt.show()

