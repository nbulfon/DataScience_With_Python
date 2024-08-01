# Agregacion de datos por categorias

import numpy as _numpy
import matplotlib.pyplot as _matplotlib
import pandas as _pandas


gender = ["Male", "Female"]
income = ["Poor", "Middle Class", "Rich"]
n = 500
gender_data = []
income_data = []

for i in range(0,500):
    gender_data.append(_numpy.random.choice(gender))
    income_data.append(_numpy.random.choice(income))
    
print(gender_data[1:10])
print(income_data[1:10])

# N(m,s) -> m + s * z   ==> Z = N(0,1)
height = 160 + 30 * _numpy.random.randn(n)
weight = 65 + 25 * _numpy.random.randn(n)
age = 30 + 12 * _numpy.random.randn(n)
income = 18000 + 3500 * _numpy.random.randn(n)

data = _pandas.DataFrame({
    "Gender": gender_data,
    "Economic Status": income_data,
    "Height" : height,
    "Weight" : weight,
    "Age" : age,
    "Income": income
    })

print(data.head())

# Agrupacion de datos

print(data.groupby("Gender"))
grouped_gender = data.groupby("Gender")

for names,groups in grouped_gender:
    print(names)
    print(groups)
    
double_group = data.groupby(["Gender","Economic Status"])
len(double_group)
for names,groups in double_group:
    print(names)
    print(groups)
# FIN Agrupacion de datos

# Operaciones sobre datos agrupados
double_group.sum()
double_group.min()

double_group.aggregate({
    "Income": _numpy.sum,
    "Age": _numpy.mean,
    "Height": _numpy.std
    })

double_group.aggregate([lambda x: _numpy.mean(x) / _numpy.std(x)])
# FIN Operaciones sobre datos agrupados


# Filtrado de datos

# filtro
double_group["Age"].filter(lambda x: x.sum() > 2400)

# transformacion de variables
zscore = lambda x : (x - x.mean()) / x.std()
z_group = double_group.transform(zscore)
_matplotlib.hist(z_group["Age"])

fill_na_mean = lambda x : x.fillna(x.mean())
double_group.transform(fill_na_mean)

# Operaciones diversas


# FIN Filtrado de datos








