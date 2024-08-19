# imports
import numpy as _numpy
import matplotlib.pyplot as _plt
from scipy import stats
import seaborn as sns; sns.set()
from sklearn.datasets import make_blobs

X, Y = make_blobs(n_samples=50, centers=2, random_state=0, cluster_std=0.6)

_plt.scatter(X[:,0], X[:,1], c = Y, s = 50, cmap="autumn")


xx = _numpy.linspace(-1, 3.5)
_plt.scatter(X[:,0], X[:,1], c = Y, s = 50, cmap="autumn")
_plt.plot([0.5], [2.1], 'x', color="blue", markeredgewidth=2, markersize=10)

for a, b in [(1,0.65), (0.5, 1.6), (-0.2, 2.9)]:
    yy = a * xx + b
    _plt.plot(xx, yy, "-k")
    
_plt.xlim(-1,3.5)



#Maximización del margen (del 'corredor'). L maximización de margén, es una técnica utilizada
# para elegir el modelo óptimo de entre todos los que están disponibles.
# (en lugar de separar con lineas rectas(algo difuso para clasificar),
# separo con corredores, margenes,pasillos).


xx = _numpy.linspace(-1, 3.5)
_plt.scatter(X[:,0], X[:,1], c = Y, s = 50, cmap="autumn")
_plt.plot([0.5], [2.1], 'x', color="blue", markeredgewidth=2, markersize=10)

for a, b, d in [(1,0.65, 0.33), (0.5, 1.6,0.55), (-0.2, 2.9, 0.2)]:
    yy = a * xx + b
    _plt.plot(xx, yy, "-k")
    # defino, entre otras cosas, la 'proyección ortogonal'.
    _plt.fill_between(xx, yy-d, yy+d, edgecolor='none', color="#BBBBBB", alpha = 0.4)
    
_plt.xlim(-1,3.5)




#Creación del modelo SVM
from sklearn.svm import SVC

model=SVC(kernel="linear", C = 1E10)
model.fit(X,Y)

# funcion Support Vector Clasifier
# el param plot_support hace refeerencia a la linea de soporte vectorial (pasillo).
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
    
    
_plt.scatter(X[:,0], X[:,1], c = Y, s = 50, cmap = "autumn")
plt_svc(model, plot_support=True)

# funcion para una Suppport Vector Machine
def plot_svm(N=10, ax=None):
    X, Y = make_blobs(
        n_samples=200,
        centers=2,
        random_state=0,
        cluster_std=0.6)
    
    # me quedo con todas las filas hasta la N-ésima.
    X = X[:N]
    Y = Y[:N]
    model = SVC(kernel="linear", C=1E10)
    model.fit(X,Y)
    
    ax = ax or _plt.gca()
    ax.scatter(X[:,0], X[:,1], c=Y, s = 50, cmap="autumn")
    
    # seteo los limites para X y para Y ->
    ax.set_xlim(-1,4)
    ax.set_ylim(-1,6)
    
    # llamo a la función de Support Vector Clasifier ->
    plt_svc(model, ax)
    

fig, ax = _plt.subplots(1,2, figsize=(16,6))
fig.subplots_adjust(left=0.0625, right = 0.95, wspace = 0.1)
for ax_i, N, in zip(ax, [60, 120]):
    plot_svm(N, ax_i)
    ax_i.set_title("N={0}".format(N))

# importo ipywidgets para hacer un widget interactivo.
from ipywidgets import interact, fixed

interact(plot_svm, N=[150, 2000], ax=fixed(None))













