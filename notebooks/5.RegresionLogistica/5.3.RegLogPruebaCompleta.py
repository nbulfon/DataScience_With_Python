# Regresión logística para predicciones bancarias

import pandas as _pandas
import numpy as _numpy
import matplotlib.pyplot as _plt
from sklearn import datasets
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

filepath = r"C:\\Nicolas\\Curso MachineLearning with Python\\datasets\\bank"
file = filepath + "\\" + "bank.csv"
data = _pandas.read_csv(file, sep=";")

# PARTE DATA CLEANING

# veo cuantos datos hay
data.shape
data.columns.values

# convirto datos de bool a int ->
#if data["y"] == "yes":
#    data["y"] = 1
#else:
#    data["y"] = 0
data["y"] = (data["y"] == "yes").astype(int)

data["education"].unique()
# voy a agrupar los nivles de educacion, para evitar tener tantas categorias
data["education"] = _numpy.where(data["education"]=="basic.4y", "Basic", data["education"])
data["education"] = _numpy.where(data["education"]=="basic.6y", "Basic", data["education"])
data["education"] = _numpy.where(data["education"]=="basic.9y", "Basic", data["education"])

data["education"] = _numpy.where(data["education"]=="high.school", "High School", data["education"])
data["education"] = _numpy.where(data["education"]=="professional.course", "Professional Course", data["education"])
data["education"] = _numpy.where(data["education"]=="university.degree", "University Degree", data["education"])

data["education"] = _numpy.where(data["education"]=="illiterate", "Illiterate", data["education"])
data["education"] = _numpy.where(data["education"]=="unknown", "Unknown", data["education"])

# FIN PARTE DATA CLEANING

# DATA WRANGLING (analisis exploratorio)
# Este analisis (sumado a pruebas de significancia, etc) me sirven para saber
# cuales variables me servirán más para mi modelo.

# veo cuantas personas compraron (y = 1) y cuantas no compraron (y = 0)
data["y"].value_counts()

# veo el valor promedio de cada columna tanto para quienes compran como para quienes no compran ->
data.groupby("y").mean()

# veo el valor promedio para las variables numericas, por educacion ->
data.groupby("education").mean()

# visualizo los datos, cruzando el nivel de educacion, con si compran o no ->
_pandas.crosstab(data.education,data.y).plot(kind="bar")
_plt.title("Frecuencia de compra en función del nivel de educación")
_plt.xlabel("Nivel de educación")
_plt.ylabel("Frecuencia de compra del producto")

# visualizo los datos, cruzando el estado civil con si compran o no
table = _pandas.crosstab(data.marital, data.y)
# hago que cada columna quede dividida por la suma de las filas
table.div(table.sum(1).astype(float),axis = 0).plot(kind="bar",stacked=True)
_plt.title("Diagrama apilado de estado civíl contra el nivel de compras")
_plt.xlabel("Estado civíl")
# en las y va la proporción de clientes porque estoy haciendo una división antes.
# Es necesario que se vean los valores relativos y no absolutos para poder comparar.
_plt.ylabel("Proporción de clientes")

# visualizo los datos, cruzando el dia de la semana con si compran o no ->
table= _pandas.crosstab(data.day_of_week, data.y)
table.div(table.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)
_plt.title("Frecuencia de compra en función del día de la semana")
_plt.xlabel("Día de la semana")
_plt.ylabel("Frecuencia de compra del producto")

# visualizo los datos, cruzando el mes con si compran o no ->
table= _pandas.crosstab(data.month, data.y)
table.div(table.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)
_plt.title("Frecuencia de compra en función del mes")
_plt.xlabel("Mes del año")
_plt.ylabel("Frecuencia de compra del producto")

# visualizo los datos, haciendo un histograma de la edad, para ver como están compuestos los clientes ->
data.age.hist()
_plt.title("Histograma de la Edad")
_plt.xlabel("Edad")
_plt.ylabel("Cliente")

# visualizo los datos, cruzando la edad con si compran o no ->
_pandas.crosstab(data.age, data.y).plot(kind="bar")

# visualizo los datos, cruzando la experiencia pasada del usuario con si compran o no ->
_pandas.crosstab(data.poutcome, data.y).plot(kind="bar")

## Conversion de las variables categórcias a Dummies



categories = ["job","marital","education",
              "housing","loan",
              "contact","month","day_of_week",
              "poutcome"]

for category in categories:
    cat_list = "cat" + "_" + category
    cat_dummies = _pandas.get_dummies(data[category], prefix=cat_list)
    data_new = data.join(cat_dummies)
    data = data_new
## FIN Conversion de las variables categórcias a Dummies
    
data_vars = data.columns.values.tolist()
to_keep = [v for v in data_vars if v not in categories]
to_keep = [v for v in data_vars if v not in ["default"]]
# ahora armo un subconjunto de los datos que me quiero quedar ->
bank_data = data[to_keep]
bank_data.columns.values

# Como la variable a predecir es la Y, voy a separar el dataset en dos.
bank_data_vars = bank_data.columns.values.tolist()
Y = ['Y']
# las X son todas las que no estén en Y.
X = [v for v in bank_data_vars if v not in Y]

## SELECCION DE RASGOS DEL MODELO

n = 12
logisticReg = LogisticRegression(solver='lbfgs',class_weight='balanced', max_iter=10000)

rfe = RFE(logisticReg, n_features_to_select=12)
rfe = rfe.fit(bank_data[X], bank_data[Y].values.ravel())

print(rfe.support_)

## FIN SELECCION DE RASGOS DEL MODELO


# FIN DATA WRANGLING




















