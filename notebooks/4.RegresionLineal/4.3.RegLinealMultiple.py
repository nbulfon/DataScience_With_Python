# imports
import pandas as _pandas
import statsmodels.formula.api as _smf
import matplotlib.pyplot as _plt
import numpy as _numpy

filepath = r"C:\\Nicolas\\Curso MachineLearning with Python\\datasets\\ads"
file = filepath + "\\" + "Advertising.csv"
data = _pandas.read_csv(file)


linearModel2 = _smf.ols(formula="Sales~TV+Newspaper",data=data).fit()
linearModel2.params
linearModel2.pvalues
linearModel2.summary()

# hago la prediccion de las ventas ->
sales_predict = linearModel2.predict(data[["TV", "Newspaper"]])

# calculo el SSD (suma de cuadrados de las diferencias) ->
SSD = sum((data["Sales"] - sales_predict) ** 2)
SSD

# calculo RSE --> acordarse de que al denomiador se le resta el nro de variables
# predictoras (k) -1 . Como k = 3, resto 2.
RSE = _numpy.sqrt(SSD / len(data)-2-1)
RSE

# calculo el error
sales_media = _numpy.mean(data["Sales"])
error = RSE / sales_media
error


## Añado el Radio al modelo a ver si mejora el R2 ->
linearModel3 = _smf.ols(formula="Sales~TV+Radio",data=data).fit()
linearModel3.summary()


# hago la prediccion de las ventas ->
sales_predict = linearModel3.predict(data[["TV", "Radio"]])

# calculo el SSD (suma de cuadrados de las diferencias) ->
SSD = sum((data["Sales"] - sales_predict) ** 2)
SSD

# calculo RSE --> acordarse de que al denomiador se le resta el nro de variables
# predictoras (k) -1 . Como k = 3, resto 2.
RSE = _numpy.sqrt(SSD / len(data)-2-1)
RSE

# calculo el error
sales_media = _numpy.mean(data["Sales"])
error = RSE / sales_media
error





















