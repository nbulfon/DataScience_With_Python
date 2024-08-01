# imports
import pandas as _pandas
import statsmodels.formula.api as _smf
import matplotlib.pyplot as _plt
import numpy as _numpy

filepath = r"C:\\Nicolas\\Curso MachineLearning with Python\\datasets\\ads"
file = filepath + "\\" + "Advertising.csv"
data = _pandas.read_csv(file)

## Añado el Radio al modelo a ver si mejora el R2 ->
linearModel3 = _smf.ols(formula="Sales~TV+Radio",data=data).fit()

#Multicolinealidad
#Factor Inflación de la Varianza

#    VIF = 1 : Las variables no están correlacionadas
#    VIF < 5 : Las variables tienen una correlación moderada y se pueden quedar en el modelo
#    VIF >5 : Las variables están altamente correlacionadas y deben desaparecer del modelo.


# Newspaper ~ TV + Radio -> R^2 VIF = 1/(1-R^2)
lm_n = _smf.ols(formula="Newspaper~TV+Radio", data = data).fit()
rsquared_newspaper = lm_n.rsquared
VIF = 1/(1-rsquared_newspaper)
VIF

# TV ~ Newspaper + Radio -> R^2 VIF = 1/(1-R^2)
lm_tv = _smf.ols(formula="TV~Newspaper+Radio", data=data).fit()
rsquared_tv = lm_tv.rsquared
VIF = 1/(1-rsquared_tv)
VIF

# Radio ~ TV + Newspaper -> R^2 VIF = 1/(1-R^2)
lm_r = _smf.ols(formula="Radio~Newspaper+TV", data=data).fit()
rsquared_radio = lm_r.rsquared
VIF = 1/(1-rsquared_radio)
VIF

linearModel3.summary()