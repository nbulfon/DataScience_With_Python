# imports

import pandas as _pandas
import numpy as _numpy
from IPython.display import display, Math, Latex

filepath = r"C:\\Nicolas\\Curso MachineLearning with Python\\datasets\\gender-purchase"
file = filepath + "\\" + "Gender Purchase.csv"
df = _pandas.read_csv(file)

# Tabla de contingencia
contingency_table = _pandas.crosstab(df["Gender"],df["Purchase"])

# sumo todos los valores del eje horizontal
contingency_table.sum(axis = 1)

# sumo todos los valores del eje vertical
contingency_table.sum(axis = 0)

# cada axis = 0 es dividida por cada axis = 1 de la tabla de contingencia
# Es decir, cada una de las filas, la divido por la suma total,
# haciendo que permanezca el formato tabla de la tabla de contingencia.
contingency_table.astype("float").div(contingency_table.sum(axis = 1), axis = 0)

# ver:
#    https://github.com/joanby/python-ml-course/blob/master/notebooks/T5%20-%201%20-%20Logistic%20Regression%20-%20Matem%C3%A1ticas-Colab.ipynb
#----------------------------
# PROBABILIDAD CONDICIONAL

# ¿cual es la probabilidad de que un cliente compre un producto sabiendo que es hombre ?
display(Math(r'P(Purchase|Male) = \frac{Numero\ total\ de\ compras\ hechas\ por\ hombres}{Numero\ total\ de\ hombres\ del\ grupo} = \frac{Purchase\cap Male}{Male}'))
121/246

0.491869918699187

display(Math(r'P(No\ Purchase|Male) = 1-P(Purchase|Male)'))
125/246

0.508130081300813

display(Math(r'P(Female|Purchase) = \frac{Numero\ total\ de\ compras\ hechas\ por\ mujeres}{Numero\ total\ de\ compras} = \frac{Female\cap Purchase}{Purchase}'))
159/280

0.5678571428571428

display(Math(r'P(Male|Purchase)'))
121/280

0.43214285714285716

display(Math(r'P(Purchase|Male)'))
print(121/246)
display(Math(r'P(NO\ Purchase|Male)'))
print(125/246)
display(Math(r'P(Purchase|Female)'))
print(159/265)
display(Math(r'P(NO\ Purchase|Female)'))
print(106/265)

0.491869918699187

0.508130081300813

0.6

0.4

#Ratio de probabilidades

#Cociente entre los casos de éxito sobre los de fracaso en el suceso estudiado y para cada grupo

display(Math(r'P_m = \ probabilidad\ de\ hacer\ compra\ sabiendo\ que\ es \ un \ hombre'))

display(Math(r'P_f = \ probabilidad\ de\ hacer\ compra\ sabiendo\ que\ es \ una\ mujer'))

# cualquier cociente de probabilidad siempre se encontrará entre 0 y +inifinito
display(Math(r'odds\in[0,+\infty]'))

display(Math(r'odds_{purchase,male} = \frac{P_m}{1-P_m} = \frac{N_{p,m}}{N_{\bar p, m}}'))

display(Math(r'odds_{purchase,female} = \frac{P_F}{1-P_F} = \frac{N_{p,f}}{N_{\bar p, f}}'))

pm = 121/246
pf = 159/265
odds_m = pm/(1-pm)# 121/125
odds_f = pf/(1-pf)# 159/106




# Si el ratio es superior a 1, es más probable el éxito que el fracaso. Cuanto mayor es el ratio, más probabilidad de éxito en nuestro suceso.
# Si el ratio es exactamente igual a 1, éxito y fracaso son equiprobables (p=0.5)
# Si el ratio es menor que 1, el fracaso es más probable que el éxito. Cuanto menor es el ratio, menor es la probabilidad de éxito del suceso.

display(Math(r'odds_{ratio} = \frac{odds_{purchase,male}}{odds_{purchase,female}}'))

























