import pandas as _pandas


data = _pandas.read_csv(r"C:\NICOLAS\Curso MachineLearning with Python\datasets\customer-churn-model\Customer Churn Model.txt")
print(data.head())
# Crear un subconjunto de datos
account_length = data["Account Length"]

subset = data[["Account Length", "Phone", "Eve Charge", "Day Calls"]]

# columnas q quiero ->
desired_columns = ["Accounts length", "VMail Message","Day Calls"]

# todas las columnas ->
all_columns_list = data.columns.values.tolist()

# me quedo con las columnas que no he quitado.
# Es decir, me quedo con el complementario
a = set(desired_columns)
b = set(all_columns_list)
sublist = b-a
sublist = list(sublist)

# imprimo desde el 10 hasta el 20
print(data[10:21])

# creo nueva columna
data["Prueba"] = data["Day Calls"] + data["Night Calls"]


















