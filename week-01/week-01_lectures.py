# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 22:06:50 2021

@author: ANalundasan
For: OMSBA 5067, Week 01 Notes
"""

import pandas as pd
import numpy as np

####################################################
dic_data = dict()
dic_data["Name"] = ["Mike", "Susan", "Eric", "Sheila"]
dic_data["dept"] = ["Math", "CS", "Phy", "CS"]
dic_data["students"] = [10, 15, 7, 19]
pd_data = pd.DataFrame(dic_data)
display(pd_data)
####################################################
## Creating a set of random numbers in the form of a 2-D array and putting them in a DataFrame with timestamps
Data_8x3 = np.random.randn(8, 3)

D = pd.date_range("20210301", periods=8)

Xdf = pd.DataFrame(Data_8x3, index=D, columns=list("XYZ"))
display(Xdf)
####################################################
## Extraction of some statistics
XDes = Xdf.describe()
display(XDes)
####################################################
## Histogram for normal distribution
mu, sigma = 0, 1
X = np.random.normal(mu, sigma, 10000)

import matplotlib.pyplot as plt
plt.hist(X, 100)