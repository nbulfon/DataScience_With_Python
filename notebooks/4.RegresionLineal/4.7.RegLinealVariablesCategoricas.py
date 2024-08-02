# imports
import pandas as _pandas
import statsmodels.formula.api as _smf
import matplotlib.pyplot as _plt
import numpy as _numpy

from sklearn.feature_selection import RFE
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model


filepath = r"C:\\Nicolas\\Curso MachineLearning with Python\\datasets\\ecom-expense"
file = filepath + "\\" + "Ecom Expense.csv"
df = _pandas.read_csv(file)

dummy_gender = _pandas.get_dummies(df["Gender"], prefix="Gender")
dummy_city_tier = _pandas.get_dummies(df["City Tier"], prefix="City")

column_names = df.columns.values.tolist()
column_names

df_new = df[column_names].join(dummy_gender)
column_names = df_new.columns.values.tolist()
df_new.head()

df_new = df_new[column_names].join(dummy_city_tier)
df_new.head()

feature_cols = ["Monthly Income", "Transaction Time", 
                "Gender_Female", "Gender_Male", 
                "City_Tier 1", "City_Tier 2", "City_Tier 3",
                "Record"]

X = df_new[feature_cols]
Y = df_new["Total Spend"]

lm = LinearRegression()
lm.fit(X,Y)

print(lm.intercept_)
print(lm.coef_)

list(zip(feature_cols, lm.coef_))
lm.score(X,Y)


df_new["prediction"] = -79.41713030137362 + df_new['Monthly Income']*0.14753898049205738 + df_new['Transaction Time']* 0.15494612549589545

SSD = _numpy.sum((df_new["prediction"] - df_new["Total Spend"])**2)
RSE = _numpy.sqrt(SSD/(len(df_new)-len(feature_cols)-1))

sales_mean=_numpy.mean(df_new["Total Spend"])
error = RSE/sales_mean




#Eliminar variables dummy redundantes

# saco var redundante, desde la primer columna, con el iloc
dummy_gender = _pandas.get_dummies(df["Gender"], prefix="Gender").iloc[:,1:]
dummy_gender.head()

# saco var redundante, desde la primer columna, con el iloc
dummy_city_tier = _pandas.get_dummies(df["City Tier"], prefix="City").iloc[:,1:]
dummy_city_tier.head()


column_names = df.columns.values.tolist()
df_new = df[column_names].join(dummy_gender)
column_names = df_new.columns.values.tolist()
df_new = df_new[column_names].join(dummy_city_tier)
df_new.head()


feature_cols = ["Monthly Income", "Transaction Time", "Gender_Male", "City_Tier 2", "City_Tier 3", "Record"]
X = df_new[feature_cols]
Y = df_new["Total Spend"]
lm = LinearRegression()
lm.fit(X,Y)

print(lm.intercept_)
list(zip(feature_cols, lm.coef_))
lm.score(X,Y)


# Transformación de variables para conseguir una relación no lineal
# (CUANDO SE ROMPE EL SUPUESTO DE LINEALIDAD EN LOS PARAMETROS O VARIABLES)

filepath = r"C:\\Nicolas\\Curso MachineLearning with Python\\datasets\\auto"
file = filepath + "\\" + "auto-mpg.csv"
data_auto = _pandas.read_csv(file)

data_auto["mpg"] = data_auto["mpg"].dropna()
data_auto["horsepower"] = data_auto["horsepower"].dropna()

_plt.plot(data_auto["horsepower"], data_auto["mpg"],"ro")
_plt.xlabel("Caballos de potencia")
_plt.ylabel("Consumo (millas por galeón)")
_plt.title("CV vs MPG")

# modelo de regresion lineal
# mpg = a + b * horsepower

# reemplazo los valores nulos con la media de las columnas ->
X = data_auto["horsepower"].fillna(data_auto["horsepower"].mean())
Y = data_auto["mpg"].fillna(data_auto["mpg"].mean())
X_data = X[:, _numpy.newaxis]

linearRegresion = LinearRegression()
# acomodo la X para cambiar el tipo de dato ->
linearRegresion.fit(X[:, _numpy.newaxis],Y) 
type(X)
type(X_data)

_plt.plot(X, Y,"ro")
_plt.plot(X, linearRegresion.predict(X_data), color="blue")
_plt.xlabel("Caballos de potencia")
_plt.ylabel("Consumo (millas por galeón)")
_plt.title("CV vs MPG")

# chequeo ->

# veo el varlor del R2
linearRegresion.score(X_data, Y)
SSD = _numpy.sum((Y -  linearRegresion.predict(X_data)) ** 2)
RSE = _numpy.sqrt(SSD / (len(X_data)-1))
Y_mean = _numpy.mean(Y)
error = RSE / Y_mean

SSD,RSE,Y_mean,error


# modelo de regresion cuadrático
# mpg = a + b * horsepower^2
X_data = X**2
X_data = X_data[:, _numpy.newaxis]

linearRegresion.fit(X_data, Y)
# veo el varlor del R2
linearRegresion.score(X_data,Y)

linearRegresion.score(X_data, Y)
SSD = _numpy.sum((Y -  linearRegresion.predict(X_data)) ** 2)
RSE = _numpy.sqrt(SSD / (len(X_data)-1))
Y_mean = _numpy.mean(Y)
error = RSE / Y_mean

SSD,RSE,Y_mean,error

# modelo de regresion lineal y cuadrático
# mpg = a + b * horsepower + c * horsepower^2

# armo un polinomio y luego ejecuto una transformmacion lineal para X_data
polinomio = PolynomialFeatures(degree=2)
X_data = polinomio.fit_transform(X[:, _numpy.newaxis])
linearRegresion = linear_model.LinearRegression()
linearRegresion.fit(X_data,Y)

# veo el varlor del R2
linearRegresion.score(X_data,Y)

# VEO LOS COEFICIENTES ->
linearRegresion.score(X_data, Y)
SSD = _numpy.sum((Y -  linearRegresion.predict(X_data)) ** 2)
RSE = _numpy.sqrt(SSD / (len(X_data)-1))
Y_mean = _numpy.mean(Y)
error = RSE / Y_mean

SSD,RSE,Y_mean,error


def regresion_validation(X_data, Y, Y_pred):
    SSD = _numpy.sum((Y - Y_pred)**2)
    RSE = _numpy.sqrt(SSD/(len(X_data)-1))
    y_mean = _numpy.mean(Y)
    error = RSE/y_mean
    print("SSD: "+str(SSD)+", RSE: " +str(RSE) + ", Y_mean: " +str(y_mean) +", error: " + str(error*100)+ "%")

for _degree in range(2,12):
    polinomio = PolynomialFeatures(degree=_degree)
    X_data = polinomio.fit_transform(X[:,_numpy.newaxis])
    linearRegresion = linear_model.LinearRegression()
    linearRegresion.fit(X_data, Y)
    print("Regresión de grado "+str(_degree))
    print("R2:" +str(lm.score(X_data, Y)))
    print(lm.intercept_)
    print(lm.coef_)
    regresion_validation(X_data, Y, lm.predict(X_data))



# El problema de los Outliers (VALORES MUY ATÍPICOS)

_plt.plot(data_auto["displacement"], data_auto["mpg"], "ro")

# armo la variable explicada y a expllicativa, y lleno los Nan con la media de la columna ->
X = data_auto["displacement"].fillna(data_auto["displacement"].mean()).to_numpy()
X = X[:,_numpy.newaxis]
Y = data_auto["mpg"].fillna(data_auto["mpg"].mean())

linearRegresion = LinearRegression()
linearRegresion.fit(X, Y)

# saco el R2
linearRegresion.score(X,Y)

_plt.plot(X,Y, "ro")
_plt.plot(X, lm.predict(X), color="blue")


#---
data_auto[(data_auto["displacement"]>250)&(data_auto["mpg"]>35)]

data_auto[(data_auto["displacement"]>300)&(data_auto["mpg"]>20)]

# saco valores muy atípicos ->
data_auto_clean = data_auto.drop([395, 258, 305, 372])

# lleno valores Nan con la media de la columna ->
X = data_auto_clean["displacement"].fillna(data_auto_clean["displacement"].mean()).to_numpy()
X = X[:,_numpy.newaxis]
# lleno valores Nan con la media de la columna ->
Y = data_auto_clean["mpg"].fillna(data_auto_clean["mpg"].mean())

linearRegresion = LinearRegression()
linearRegresion.fit(X, Y)

# saco R2 ->
linearRegresion.score(X,Y)

# grafico
_plt.plot(X,Y, "ro")
_plt.plot(X, linearRegresion.predict(X), color="green")







