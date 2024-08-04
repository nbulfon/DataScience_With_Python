## DEFINIR LA FUNCION DE ENTORNO L(b)

from IPython.display import display, Math, Latex


def likelihood(y, pi):
    import numpy as _numpy
    total_sum = 1
    sum_in = list(range(1, len(y)+1))
    for i in range(len(y)):
        sum_in[i] = _numpy.where(y[i] == 1, pi[i], 1-pi[i])
        total_sum = total_sum * sum_in[i]
    return total_sum

def logitprobs(X,beta):
    import numpy as _numpy
    
    n_rows = _numpy.shape(X)[0]
    n_cols = _numpy.shape(X)[1]
    pi=list(range(1,n_rows+1))
    expon=list(range(1,n_rows+1))
    for i in range(n_cols):
        expon[i] = 0
        for j in range(n_cols):
            ex = X[i][j] * beta[j]
            expon[i] = ex + expon[i]
        with _numpy.errstate(divide="ignore",invalid="ignore"):
            pi[i] = 1 / (1+ _numpy.exp(-expon[i]))
    return pi


# Calcular la matriz diagonal W
def findW(pi):
    import numpy as _numpy
    # creo la matriz
    n = len(pi)
    # matriz de nXn  de n filas por n columnas
    W = _numpy.zeros(n*n).reshape(n,n)
    for i in range(n):
        print(i)
        W[i,i] = pi[i] * (1-pi[i])
        W[i,i].astype("float")
    return W

# Obtener la solución de la función logística
def logistics(X,Y, limit):
    import numpy as _numpy
    from numpy import linalg
    n_row = _numpy.shape(X)[0]
    bias = _numpy.ones(n_row).reshape(n_row,1)
    X_new = _numpy.append(X,bias,axis=1)
    n_col = _numpy.shape(X)[1]
    beta = _numpy.zeros(n_col).reshape(n_col,1)
    # defino el incremento de las raices del sistema (root_dif)
    root_dif = _numpy.array(range(1,n_col+1)).reshape(n_col,1)
    iter_i = 10000
    while iter_i > limit:
        print(str(iter_i)+","+str(limit))
        # llamo al metodo para obtener las probabilidades con estos X y estos betas
        pi = logitprobs(X_new, beta)
        print(pi)
        # llamo al metodo para obtener la matriz W con estas probabilidades
        W = findW(pi)
        print(W)
        # transpongo la matriz porque necesito tener vectores columna...y todos me vienen como vectores fila.
        numerador = (_numpy.transpose(_numpy.matrix(X_new))*_numpy.matrix(Y-_numpy.transpose(pi)).transpose())
        denominador = (_numpy.matrix(_numpy.transpose(X_new))*_numpy.matrix(W)*_numpy.matrix(X_new))
        root_dif = _numpy.array(linalg.inv(denominador)*numerador)
        beta = beta + root_dif
        print(beta)
        # la iteracion iesima, es la suma del cuadrado de las diferencias.
        # Analizo cuanto movimiento hubo de un momento al siguiente. Es decir, voy a seguir iterando
        # mientras haya algo de cambio.
        iter_i = _numpy.sum(root_dif*root_dif)
        print(iter_i)
        ll = likelihood(Y, pi)
    return beta


## Parte COMPROBACION EXPERIMENTAL

import numpy as _numpy

# creo un vector experimental
X = _numpy.array(range(10)).reshape(10,1)

# las Y las clasifico a mano, porque el vector Y debe ser un clasificador manual (binario)
Y = [0,0,0,0,1,0,1,0,1,1]

# agrego datos a las X
bias = _numpy.ones(10).reshape(10,1)
X_new = _numpy.append(X, bias,axis=1)

a = logistics(X, Y, 0.00001)
## FIN COMPROBACION EXPERIMENTAL

import statsmodels.api as _sm

logit_model = _sm.Logit(Y, X_new)
result = logit_model.fit()
        
print(result.summary())
        
        
        
        
        
        