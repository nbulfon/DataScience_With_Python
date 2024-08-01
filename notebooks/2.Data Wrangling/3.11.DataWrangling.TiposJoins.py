#imports
from IPython.display import Image
import numpy as np
#imports
import numpy as _numpy
import matplotlib.pyplot as _mathplotlib
import pandas as _pandas

filepath = r"C:\\NICOLAS\\Curso MachineLearning with Python\\datasets\\athletes"
file = filepath + "\\" + "Medals.csv"
data_main = _pandas.read_csv(file, encoding="ISO-8859-1")

#Inner Join <= A (Left Join), B (Right Join) <= Outer Join

out_athletes = np.random.choice(data_main["Athlete"], size = 6, replace = False)

data_country_dlt = data_country_dp[(~data_country_dp["Athlete"].isin(out_athletes)) & 
                                   (data_country_dp["Athlete"] != "Michael Phelps")]

data_sports_dlt = data_sports_dp[(~data_sports_dp["Athlete"].isin(out_athletes)) &
                                (data_sports_dp["Athlete"] != "Michael Phelps")]

data_main_dlt = data_main[(~data_main["Athlete"].isin(out_athletes)) & 
                         (data_main["Athlete"] != "Michael Phelps")]