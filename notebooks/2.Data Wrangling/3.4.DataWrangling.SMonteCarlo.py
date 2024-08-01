# Simulación de Monte Carlo

import numpy as _numpy
import matplotlib.pyplot as _matplotlib


# generamos 2 numeros aleatorios uniforme x e y entre 0 y 1, en total 1000 veces.

# calcularemos x^2 + y^2
#* Si el valor es inferior a 1 -> estamos dentro del circulo
#* Si el valor es superior a 1 -> estamos fuera del circulo
# Entonces, calculamos el número total de veces que están dentro del circulo y lo dividimos entre el número total
# de intentos para obtener una aproximación de la probabilidad de caer dentro del circulo
# (CasosFavorables/CasosPosibles).
# Usamos dicha probabilidad para aproximar el valor de pi.
# Repetimos el experimiento un número suficiente de veces (por ejemplo, 100)
# para obtener 100 diferentes aproximaciones de pi.
# Calculamos el promedio de los 100 experimentos anteriores para dar un valor final a pi.

pi_avg = 0
n = 10000
pi_value_list = []
for i in range(100):
    value = 0
    x = _numpy.random.uniform(0,1,n).tolist()
    y = _numpy.random.uniform(0,1,n).tolist()
    for j in range(n):
        z = _numpy.sqrt(x[j] + y[j] * y[j])
        if (z <= 1):
            value += 1
    # Convierto el dato a float para captar los decimales
    float_value = float(value)
    # CasosFavorables/ CasosPosibles
    pi_value = float_value * 4 / n
    pi_value_list.append(pi_value)
    pi_avg += pi_value

pi = pi_avg / 100
print(pi)
_matplotlib.plot(pi_value_list)













