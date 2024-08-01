# Carga de datos
#%config IPCompleter.greedy=True

import pandas as _pandas

# paths
mainPath = r"C:\NICOLAS\Curso MachineLearning with Python\datasets"
fileName = r"\titanic\titanic3.csv"
fullpath = mainPath + fileName

data = _pandas.read_csv(fullpath)

#----
data2 = _pandas.read_csv(mainPath + r"\customer-churn-model\Customer Churn Model.txt")

data_cols = _pandas.read_csv(mainPath + r"\customer-churn-model\Customer Churn Model.csv")
data_col_list = data_cols["Column_Names"].tolist()

data2 = _pandas.read_csv(mainPath + r"\customer-churn-model\Customer Churn Model.txt",header=None, names= data_col_list)
#----

# Manera alternativa al _pandas.read_csv() -->
data3 = open(mainPath + r"\customer-churn-model\Customer Churn Model.txt","r")
cols = data3.readline().strip().split(",")
n_cols = len(cols)

counter = 0
main_dict = {}
for col in cols:
    main_dict[col] = []
    
for line in data3:
    values = line.strip().split(",")
    for i in range(len(cols)):
        main_dict[cols[i]].append(values[i])
    counter += 1
    
print("El data set tiene %d filas y %d columnas"%(counter, n_cols))
  
df3 = _pandas.DataFrame(main_dict)
#------ FIN Manera alternativa al _pandas.read_csv().

infile = mainPath + r"\customer-churn-model\Customer Churn Model.txt"
outfile = mainPath + r"\customer-churn-model\Tab Customer Churn Model.txt"

with open(infile, "r") as infile1:
    with open(outfile, "w") as outfile1:
        for line in infile1:
            fields = line.strip().split(",")
            outfile1.write("\t".join(fields))
            outfile1.write("\n")
            
df4 = _pandas.read_csv(outfile, sep= "\t")
#--------

# Leer datos desde URL externa ->
 
#opcion 1   
medals_url = "https://winterolympicsmedals.com/medals.csv"
medals_data = _pandas.read_csv(medals_url)

#opcion 2
import urllib3
http = urllib3.PoolManager()
r = http.request('GET', medals_url)
r.status
response = r.data

import csv
cr = csv.reader(response)

for row in cr:
    print(row)
# FIN Leer datos desde URL externa ->
       
    
mainpath = r"\content\drive\My Drive\Curso Machine Learning con Python\datasets"
filename = r"\titanic\titanic3.xls"    
    
titanic2 = _pandas.read_excel(mainpath + filename, "titanic3") 
    
filename = r"\titanic\titanic3.xlsx"
titanic3 = _pandas.read_excel(mainpath + filename, "titanic3")   
    
titanic3.to_csv(mainpath + "\titanic\titanic_custom.csv")
    
titanic3.to_excel(mainpath + "\titanic\titanic_custom.xls")    
    
titanic3.to_json(mainpath + "\titanic\titanic_custom.json")
    




















    