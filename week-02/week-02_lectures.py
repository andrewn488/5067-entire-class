# -*- coding: utf-8 -*-
"""
Created on Sat Apr  3 22:20:44 2021

@author: ANalundasan
For: OMSBA 5067, Week 02 Notes
"""

import random
import math

theta = 0.3

all_est = []
for i in range(0, 10000):
    ones = 0
    zeros = 0
    for j in range (0, 100):
        p = random.random()
        if p <= theta:
            ones = ones + 1
        else:
            zeros = zeros + 1
    est_theta = ones / (ones + zeros)
    all_est.append(est_theta)
    
import matplotlib.pyplot as plt
plt.hist(all_est)
plt.show()

import statistics
var_est = statistics.variance(all_est)
mean_est = statistics.mean(all_est)

## GaussianNB

import numpy as np
X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
Y = np.array([1, 1, 1, 2, 2, 2])
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X, Y)       ## USED FOR TRAINING
print(gnb.predict([[-0.8, 1]]))     ## USED FOR PREDICTION (e.g. in testing)

## MultinomialNB
X = np.random.randint(5, size=(6, 100))
y = np.array([1, 2, 3, 4, 5, 6])
from sklearn.naive_bayes import CategoricalNB
clf = CategoricalNB()
clf.fit(X, y)
print(clf.predict(X[2:4]))

################## GAUSSIAN ACTIVITY #########################################


## P(X = 2 | C = R)
x_1 = 2
mu_1 = 1.8
sigma2_1 = 4.96
px_1 = 1 / (math.sqrt(2 * math.pi * sigma2_1)) * (math.e ** ( -1 * ((x_1 - mu_1) ** 2) / (2 * sigma2_1)))
print('P(X = 2 | C = R) = ', px_1)

## P(X = 2 | C = B)
x_2 = 2
mu_2 = 1.5
sigma2_2 = 1.25
px_2 = 1 / (math.sqrt(2 * math.pi * sigma2_2)) * (math.e ** (-1 * ((x_2 - mu_2) ** 2) / (2 * sigma2_2)))
print('P(X = 2 | C = B) = ', px_2)

## P(Y = 2 | C = R)
y_1 = 2
mu_3 = 1.8
sigma2_3 = 2.56
py_1 = 1 / (math.sqrt(2 * math.pi * sigma2_3)) * (math.e ** ( -1 * ((x_1 - mu_3) ** 2) / (2 * sigma2_3)))
print('P(Y = 2 | C = R) = ', py_1)

## P(Y = 2 | C = B)
y_1 = 2
mu_4 = 0.25
sigma2_4 = 0.188
py_1 = 1 / (math.sqrt(2 * math.pi * sigma2_4)) * (math.e ** ( -1 * ((x_1 - mu_4) ** 2) / (2 * sigma2_4)))
print('P(Y = 2 | C = B) = ', py_1)