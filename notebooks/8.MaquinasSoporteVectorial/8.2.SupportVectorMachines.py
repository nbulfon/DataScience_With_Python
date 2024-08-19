# imports
import numpy as _numpy
import matplotlib.pyplot as _plt
from scipy import stats
import seaborn as sns; sns.set()
from sklearn.datasets.samples_generator import make_blobs

X, Y = make_blobs(n_samples=50, centers=2, random_state=0, cluster_std=0.6)

_plt.scatter(X[:,0], X[:,1], c = Y, s = 50, cmap="autumn")


xx = _numpy.linspace(-1, 3.5)
_plt.scatter(X[:,0], X[:,1], c = Y, s = 50, cmap="autumn")
_plt.plot([0.5], [2.1], 'x', color="blue", markeredgewidth=2, markersize=10)

for a, b in [(1,0.65), (0.5, 1.6), (-0.2, 2.9)]:
    yy = a * xx + b
    _plt.plot(xx, yy, "-k")
    
_plt.xlim(-1,3.5)



#Maximización del margen

xx = _numpy.linspace(-1, 3.5)
_plt.scatter(X[:,0], X[:,1], c = Y, s = 50, cmap="autumn")
_plt.plot([0.5], [2.1], 'x', color="blue", markeredgewidth=2, markersize=10)

for a, b, d in [(1,0.65, 0.33), (0.5, 1.6,0.55), (-0.2, 2.9, 0.2)]:
    yy = a * xx + b
    _plt.plot(xx, yy, "-k")
    _plt.fill_between(xx, yy-d, yy+d, edgecolor='none', color="#BBBBBB", alpha = 0.4)
    
_plt.xlim(-1,3.5)




#Creación del modelo SVM


















