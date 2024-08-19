# Kernels no-lineales
## Identificar fronteras no-lineales.
# (se usa cuando tenemos una distribución de puntos, en los cuales no existe ninguna
# linea, ninguna recta, capaz de separarlos)

# imports
from sklearn.datasets import make_circles, make_blobs
import matplotlib.pyplot as _plt
import numpy as _numpy
from sklearn.svm import SVC

X,Y = make_circles(100,factor = .1, noise = .1)

def plt_svc(model, ax=None, plot_support=True):
    """Plot de la función de decisión para una clasificación en 2D con SVC"""
    if ax is None:
        ax = _plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    ##Generamos la parrila de puntos para evaluar el modelo
    xx = _numpy.linspace(xlim[0], xlim[1], 30)
    yy = _numpy.linspace(ylim[0], ylim[1], 30)
    Y, X = _numpy.meshgrid(yy,xx)
    
    xy = _numpy.vstack([X.ravel(), Y.ravel()]).T
    P = model.decision_function(xy).reshape(X.shape)

    
    ##Representamos las fronteras y los márgenes del SVC
    ax.contour(X,
               Y,
               P,
               colors="k",
               levels=[-1,0,1],
               alpha = 0.5,
               linestyles=["--", "-", "--"])
    
    print(model.support_vectors_)
    
    if plot_support:
        ax.scatter(model.support_vectors_[:,0], 
                   model.support_vectors_[:,1], 
                   s=300, linewidth=1, facecolors = "black");
    
    
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)


_plt.scatter(X[:,0],X[:,1], c=Y, s=50, cmap="autumn")
plt_svc(SVC(kernel="linear").fit(X,Y),plot_support=False)

# para comprobar, voy a elevar la dimensión del modelo, y establecer un separador
# aleatorio r.
# factor r con dimensión de campana de Gauss.
r = _numpy.exp(-(X**2).sum(1))
r

from mpl_toolkits import mplot3d

def plot_3D(elevacion=30, azim=30,X=X,Y=Y,r=r):
    ax = _plt.subplot(projection="3d")
    ax.scatter3D(X[:,0], X[:,1],r,c=Y,s=50,cmap="autumn")
    ax.view_init(elev=elevacion, azim=azim)
    
    ax.set_xlabel("X[0]")
    ax.set_ylabel("X[1]")
    ax.set_zlabel("r")

# importo ipywidgets para hacer un widget interactivo.
from ipywidgets import interact, fixed

interact(plot_3D, elev=[-90,-60,-30,0,30,60,90], 
         azim=[-180,-150,-120,-90,-60,-30,0,30,60,90,120,150, 180], 
         X = fixed(X), Y = fixed(Y), r = fixed(r))


# RADIAL BASIC FUNCTIONS
# se usa para graficar y representar modelos no-lineales.
# (kernel en este caso no es lineal, sino que es RADIAL...usa una elipse para graficar).


rbf = SVC(kernel="rbf", C=1E6)
rbf.fit(X,Y)

_plt.scatter(X[:,0], X[:,1], c=Y, s=50, cmap="autumn")
plt_svc(rbf)
_plt.scatter(rbf.support_vectors_[:,0],
             rbf.support_vectors_[:,1],
             s=300,
             lw=1,
             facecolors="blue")
# FIN RADIAL BASIC FUNCTIONS


# Ajustar los parámetros de SVM
# tb. acá, es para separar modelos de manera no-lineal.
# (cuando los datos no pueden separarse de manera lineal.)
X, Y = make_blobs(n_samples=100, centers = 2, random_state=0, cluster_std=1.2)

_plt.scatter(X[:,0], X[:,1], c = Y, s=50, cmap="autumn")
model = SVC(kernel="linear", C=10)
model.fit(X,Y)
plt_svc(model)

# notar que acá le relajo la desviación estándar...
X, Y = make_blobs(n_samples=100, centers=2, random_state=0, cluster_std=0.8)
_plt.scatter(X[:,0], X[:,1], c = Y, s=50, cmap="autumn")

fig, ax = _plt.subplots(1,2, figsize=(16,6))
fig.subplots_adjust(left = 0.05, right = 0.95, wspace=0.1)

for ax_i, C in zip(ax, [10.0, 0.1]):
    model = SVC(kernel="linear", C=C)
    model.fit(X,Y)
    ax_i.scatter(X[:,0],X[:,1], c = Y, s = 50, cmap="autumn")
    plt_svc(model, ax_i)
    ax_i.set_title("C = {0:.1f}".format(C), size = 15)
# FIN Ajustar los parámetros de SVM














