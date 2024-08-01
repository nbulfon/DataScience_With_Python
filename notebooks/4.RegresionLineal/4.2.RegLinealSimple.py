# imports
import pandas as _pandas
import statsmodels.formula.api as _smf
import matplotlib.pyplot as _plt
import numpy as _numpy

filepath = r"C:\\Nicolas\\DataScience_With_Python\\datasets\\ads"
file = filepath + "\\" + "Advertising.csv"
data = _pandas.read_csv(file)


linearModel = _smf.ols(formula="Sales~TV",data=data).fit()
print(linearModel.params)
print(linearModel.pvalues)
print(linearModel.rsquared_adj)

# el modelo lineal predictivo seria = 7.032594 + 0.047537 * TV
linearModel.summary()

sales_prediccion = linearModel.predict(_pandas.DataFrame(data["TV"]))
sales_prediccion

data.plot(kind = "scatter", x = "TV", y = "Sales")
df = _pandas.DataFrame(data["TV"])
_plt.plot(df, sales_prediccion, color="red", linewidth=2)

data["sales_prediccion"] =7.032594 +0.047537 * data["TV"]
data["RSE"] = (data["Sales"] -data["sales_prediccion"])**2
SSD = sum(data["RSE"])
RSE = _numpy.sqrt(SSD/(len(data)-2))
RSE

sales_media = _numpy.mean(data["Sales"])
error = RSE/sales_media

_plt.hist((data["Sales"]-data["sales_prediccion"]))