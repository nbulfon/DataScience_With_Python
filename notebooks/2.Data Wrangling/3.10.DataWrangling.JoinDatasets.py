#imports
import numpy as _numpy
import matplotlib.pyplot as _mathplotlib
import pandas as _pandas

filepath = r"C:\\NICOLAS\\Curso MachineLearning with Python\\datasets\\athletes"
file = filepath + "\\" + "Medals.csv"
data_main = _pandas.read_csv(file, encoding="ISO-8859-1")

# veo cuantos estan repetidos por atleta
a = data_main["Athlete"].unique().tolist()
len(a)

data_country = _pandas.read_csv(filepath + "\\Athelete_Country_Map.csv", encoding="ISO-8859-1")
data_argentina = data_country[data_country["Country"] == "Argentina"]
print(len(data_argentina))

data_sports = _pandas.read_csv(filepath + "\\Athelete_Sports_Map.csv", encoding="ISO-8859-1")

print(
      data_sports[
          (data_sports["Athlete"] == "Cheng Jing") |
          (data_sports["Athlete"] == "Ángel Di María") |
          (data_sports["Athlete"] == "Matt Ryan")
          ]
      )

# JOINS

data_country_dp = data_country.drop_duplicates(subset="Athlete")
print(len(data_country_dp) == len(a))

data_main_country = _pandas.merge(
    left= data_main,
    right = data_country,
    left_on="Athlete",
    right_on="Athlete"
    )

tamanioTotal = len(data_main) + len(data_country)
print(len(data_main_country) == tamanioTotal)

data_sports_dp = data_sports.drop_duplicates(subset="Athlete")
print(len(data_sports_dp) == len(a))

data_final = _pandas.merge(
    left = data_main_country,
    right = data_sports_dp,
    left_on="Athlete",
    right_on="Athlete"
 )
data_final.shape
# FIN JOINS

















