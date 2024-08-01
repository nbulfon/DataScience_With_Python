# Resumen de los datos, dimensiones y estructuras

import pandas as _pandas

# paths
mainPath = r"C:\NICOLAS\Curso MachineLearning with Python\datasets"
fileName = r"\titanic\titanic3.csv"
fullpath = mainPath + fileName

urldata = "https://raw.githubusercontent.com/joanby/python-ml-course/master/datasets/titanic/titanic3.csv"

data = _pandas.read_csv(urldata)

print(data.head(10))

#obtengo la dimension ->
dimension = data.shape
print(data.columns.values)

# vamos a hacer un resumen de los estadisticos basicos de las variables numericas ->

print(data.describe())
print(data.dtypes)


# Missing values ->

print('¿faltan valores?',_pandas.isnull(data["body"]))

# los valores que faltan en un data set pueden deberse a:
    #- Extraccion de los datos
    #- Recoleccion de los datos
# FIN Missing values


### Borrado de valores faltantes ->

# axis = 0 borra toda la fila donde hayan valores nulos
# axis = 1 borra toda la columna donde hayan valores nulos
data.dropna(axis=0, how="all")

data2 = data.copy()
data2.dropna(axis=0, how="any")
### FIN Borrado de valores faltantes

### Metodo de Computo de los valores faltantes (cambiar los valores q falten por otros valores)- >
data3 = data.copy()
data3 = data3.fillna("Desconocido")
print(data3)

data4 = data.copy()
data4["body"].fillna("Desconocido",inplace=True)

print(_pandas.isnull(data4["age"]).values.ravel().sum())

### Metodo rellenar valores nulos con el promedio de la columna ->
data4["age"].fillna(data["age"].mean(), inplace=True)
### FIN Metodo rellenar valores nulos con el promedio de la columna

### FIN Metodo de Computo de los valores faltantes

# Variables Dummies ->
dummy_sex = _pandas.get_dummies(data["sex"],prefix="sex")
print(dummy_sex.head(10))

column_name = data.columns.values.tolist()
print(column_name)

# axis = 1 para que se dropee la columna sex
data = data.drop(["sex"],axis=1)

_pandas.concat( [data, dummy_sex], axis=1)


#funcion createDmmies
def createDummies(df, var_name):
    dummy = _pandas.get_dummies(df[var_name],prefix=var_name)
    df = df.drop(var_name,axis =1)
    df = _pandas.concat([df,dummy],axis=1)
    return df
# Fin funcion createDummies

createDummies(data3, "sex")
    
# FIN Variables Dummies





























