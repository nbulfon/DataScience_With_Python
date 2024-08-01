import numpy as _numpy
import random as _random


# genero un numero aleatorio entre 1 y 100.
# La forma mas clasica es enre 0 y 1
_numpy.random.randint(1,100)

# genero un numero aleatorio entre 0 y 1
_numpy.random.random()

# función que genera una lista de números aleatorios enteros dentro del intervalo [a,b]
def randint_list(n,a,b):
    x = []
    for i in range(n):
        x.append(_numpy.random.randint(a,b))
    return x

# llamo a la funcion
randint_list(2, 1, 50)

# genero nros del 1 al 100, multiplos de 7 ->
_random.randrange(1,100,7)

# Shuffling (mezclar) ->
a = _numpy.arange(100)
_numpy.random.shuffle(a)
 
#------------------------------
# Semilla de la generación aleatoria
# seed = semilla
# Establecer una "semilla" es super importante para la reproductividad del experimento aleatorio.
_numpy.random.seed(2018)
for i in range(5):
    print(_numpy.random.random())
# FIN Semilla de la generación aleatoria


















