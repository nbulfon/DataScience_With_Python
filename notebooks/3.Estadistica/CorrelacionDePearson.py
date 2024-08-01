# imports
import pandas as _pandas
import numpy as _numpy
import matplotlib.pyplot as _plt


filepath = r"C:\\NICOLAS\\Curso MachineLearning with Python\\datasets\\ads"
file = filepath + "\\" + "Advertising.csv"
data_ads = _pandas.read_csv(file)

data_ads["corrn"] = (data_ads["TV"] - _numpy.mean(data_ads["TV"])) *(data_ads["Sales"] - _numpy.mean(data_ads["Sales"]))
data_ads.head()

data_ads["corrn1"] = (data_ads["TV"] - _numpy.mean(data_ads["TV"])) **2
data_ads.head()

data_ads["corrn2"] = (data_ads["Sales"] - _numpy.mean(data_ads["Sales"])) **2
data_ads.head()


corr_pearson = sum(data_ads["corrn"]) / _numpy.sqrt(sum(data_ads["corrn1"]) *sum(data_ads["corrn2"]))

def corr_coeff(df, var1, var2):
    df["corrn"] = (df[var1] - _numpy.mean(data_ads[var1])) *(df[var2] - _numpy.mean(df[var2]))

    df["corrn1"] = (data_ads[var1] - _numpy.mean(data_ads[var1])) **2

    df["corrn2"] = (data_ads[var2] - _numpy.mean(data_ads[var2])) **2
    corr_pearson = sum(df["corrn"]) / _numpy.sqrt(sum(df["corrn1"]) *sum(df["corrn2"]))
    return corr_pearson


corr_coeff(data_ads,"TV","Sales")

cols = data_ads.columns.values
for x in cols:
    for y in cols:
        print(x + ", "+ y + ", " + ":"+ str(corr_coeff(data_ads,x,y)))

_plt.plot(data_ads["TV"], data_ads["Sales"], "ro")
_plt.title("Gastos en TV vs Ventas del producto")










