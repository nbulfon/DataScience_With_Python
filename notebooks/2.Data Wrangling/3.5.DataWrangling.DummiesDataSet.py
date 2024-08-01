# Dummies data sets

import numpy as _numpy
import matplotlib.pyplot as _matplotlib
import pandas as _pandas

data = _pandas.DataFrame({
        'A' : _numpy.random.randn(10),
        'B': 1.5 + 2.5 * _numpy.random.randn(10),
        'C': _numpy.random.uniform(5,32,10)
    })

data = _pandas.read_csv(r"C:\NICOLAS\Curso MachineLearning with Python\datasets\customer-churn-model\Customer Churn Model.txt")
column_names = data.columns.values.tolist()
a = len(column_names)
new_data = _pandas.DataFrame(
    {
     'Column Name': column_names,
     'A': _numpy.random.randn(a),
     'B':_numpy.random.uniform(0,1,a)
    }, index= range(42,42+a))