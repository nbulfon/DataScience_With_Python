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
Y = ['y']
# las X son todas las que no estén en Y.
X = [v for v in bank_data_vars if v not in Y]

## SELECCION DE RASGOS DEL MODELO

n = 12
logisticReg = LogisticRegression(solver='lbfgs',class_weight='balanced', max_iter=10000)

rfe = RFE(logisticReg, n_features_to_select=12)
rfe = rfe.fit(bank_data[X], bank_data[Y].values.ravel())

print(rfe.support_)

print(rfe.ranking_)

z=zip(bank_data_vars,rfe.support_, rfe.ranking_)

list(z)


# creo mi array de columnas definitivas ->
cols = ["previous", "euribor3m", "job_blue-collar", "job_retired", "month_aug", "month_dec", 
        "month_jul", "month_jun", "month_mar", "month_nov", "day_of_week_wed", "poutcome_nonexistent"]
X = bank_data[cols]
Y = bank_data["y"]

## FIN SELECCION DE RASGOS DEL MODELO
# FIN DATA WRANGLING


# Implementacion del modelo en Python con statsmodel.api
import statsmodels.api as sm

logit_model = sm.Logit(Y, X)

result = logit_model.fit()

result.summary()


# Implementación del modelo en Python con scikit-learn

from sklearn import linear_model

logit_model = linear_model.LogisticRegression()
logit_model.fit(X,Y)

logit_model.score(X,Y)

1-Y.mean()  

_pandas.DataFrame(list(zip(X.columns, _numpy.transpose(logit_model.coef_))))

# Validación del modelo logístico
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.3, random_state=0)

lm = linear_model.LogisticRegression()
lm.fit(X_train, Y_train)



from IPython.display import display, Math, Latex

display(Math(r'Y_p=\begin{cases}0& si\ p\leq0.5\\1&si\ p >0.5\end{cases}'))

probs = lm.predict_proba(X_test)

probs

prediction = lm.predict(X_test)

prediction

  
prob = probs[:,1]
prob_df = _pandas.DataFrame(prob)
threshold = 0.1
prob_df["prediction"] = _numpy.where(prob_df[0]>threshold, 1, 0)
prob_df.head()


_pandas.crosstab(prob_df.prediction, columns="count")

390/len(prob_df)*100

threshold = 0.15
prob_df["prediction"] = _numpy.where(prob_df[0]>threshold, 1, 0)
_pandas.crosstab(prob_df.prediction, columns="count")


331/len(prob_df)*100


threshold = 0.05
prob_df["prediction"] = _numpy.where(prob_df[0]>threshold, 1, 0)
_pandas.crosstab(prob_df.prediction, columns="count")


from sklearn import metrics

metrics.accuracy_score(Y_test, prediction)


# Validación cruzada
# la validacion cruzada se requiere en muchos casos para que en el modelo no haya problemas
# de OverFitting (anda perfecto para un conjunto de entrenamiento, y mal para el de prueba).

from sklearn.model_selection import cross_val_score

scores = cross_val_score(linear_model.LogisticRegression(), X, Y, scoring="accuracy", cv=10)

scores

scores.mean()

# Matrices de Confusión y curvas ROC
# ROC (Caracteristicas Operativas del Receptor)

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.3, random_state=0)

lm = linear_model.LogisticRegression()
lm.fit(X_train, Y_train)

probs = lm.predict_proba(X_test)

prob=probs[:,1]
prob_df = _pandas.DataFrame(prob)
threshold = 0.1 # esto es el nivel de sensibilidad elegido.
prob_df["prediction"] = _numpy.where(prob_df[0]>=threshold, 1, 0)
prob_df["actual"] = list(Y_test)
prob_df.head()

confusion_matrix = pd.crosstab(prob_df.prediction, prob_df.actual)

TN=confusion_matrix[0][0]
TP=confusion_matrix[1][1]
FN=confusion_matrix[0][1]
FP=confusion_matrix[1][0]

sens = TP/(TP+FN)
sens

espc_1 = 1-TN/(TN+FP)
espc_1

thresholds = [0.04, 0.05, 0.07, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.3, 0.4, 0.5]
sensitivities = [1]
especifities_1 = [1]

for t in thresholds:
    prob_df["prediction"] = _numpy.where(prob_df[0]>=t, 1, 0)
    prob_df["actual"] = list(Y_test)
    prob_df.head()

    confusion_matrix = _pandas.crosstab(prob_df.prediction, prob_df.actual)
    TN=confusion_matrix[0][0]
    TP=confusion_matrix[1][1]
    FP=confusion_matrix[0][1]
    FN=confusion_matrix[1][0]
    
    sens = TP/(TP+FN)
    sensitivities.append(sens)
    espc_1 = 1-TN/(TN+FP)
    especifities_1.append(espc_1)

sensitivities.append(0)
especifities_1.append(0)

import matplotlib.pyplot as plt

#%matplotlib inline
plt.plot(especifities_1, sensitivities, marker="o", linestyle="--", color="r")
x=[i*0.01 for i in range(100)]
y=[i*0.01 for i in range(100)]
plt.plot(x,y)
plt.xlabel("1-Especifidad")
plt.ylabel("Sensibilidad")
plt.title("Curva ROC")

#HAY QUE ESPERAR QUE ACTUALICE GGPLOT LAS LIBRERIAS, SINO HAY QUE MODIFICAR ARCHIVOS INTERNOS

from sklearn import metrics
from pandas import Timestamp
from ggplot import *
espc_1, sensit, _ = metrics.roc_curve(Y_test, prob)

df = pd.DataFrame({
    "esp":espc_1,
    "sens":sensit
})

#ggplot(df, aes(x="esp", y="sens")) +geom_line() + geom_abline(linetype="dashed")+xlim(-0.01,1.01)+ylim(-0.01,1.01)+xlab("1-Especifid
auc = metrics.auc(espc_1, sensit)
auc
ggplot(df, aes(x="esp", y="sens")) + geom_area(alpha=0.25)+geom_line(aes(y="sens"))+ggtitle("Curva ROC y AUC=%s"%str(auc))
# FIN Matrices de Confusión y curvas ROC

















