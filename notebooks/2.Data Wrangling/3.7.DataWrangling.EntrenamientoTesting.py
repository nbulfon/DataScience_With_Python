# Conjunto de entrenamiento y conjunto de testing

import numpy as _numpy
import matplotlib.pyplot as _mathplotlib
import pandas as _pandas
import sklearn as _sklearn
from sklearn.model_selection import train_test_split


data = _pandas.read_csv(r"C:\NICOLAS\Curso MachineLearning with Python\datasets\customer-churn-model\Customer Churn Model.txt")
len(data)

# MANERA 1

# dividir utilizando la distribución normal ->
a = _numpy.random.randn(len(data))
_mathplotlib.hist(a)

check = (a < 0.8) # 0.8 porque 80% se usa para entrenar al modelo
_mathplotlib.hist(check.astype(int))

# el conjunto de training seran los datos que  cumplen el check (80% para entrenar)
training = data[check]
# el conjunto de testing seran los que no lo cumplen (20% para testear)
testing =data[~check]

len(training)
len(testing)
# FIN MANERA 1

# MANERA 2 (sklearn)

# defino los conjuntos
conjuntoTrain, conjuntoTest = train_test_split(data, test_size=0.2)
len(conjuntoTrain)
len(conjuntoTest)
# FIN MANERA 2 (sklearn)


# LA MANERA 3 ES LA MEJOR MANERA...MAS "CASERA"
# MANERA 3 Usando una función de shuffle (de mezclado de datos)

# al poner una semilla al principio puedo tener reproductividad.
data = _sklearn.utils.shuffle(data)

cut_id = int(0.75 * len(data))
# defino los conjuntos
train_data = data[:cut_id]
test_data = data[cut_id+1:]

len(train_data)
len(test_data)
# FIN MANERA 3 Usando una función de shuffle (de mezclado de datos)




