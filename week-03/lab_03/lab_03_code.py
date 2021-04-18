# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 22:15:28 2021

@author: ANalundasan
For: OMSBA 5067, Lab 3
"""
import numpy as np
import matplotlib.pyplot as plt
import math

#################### STEP 1 - KNN Classifier #################################
data = np.array([ [1, 1,1,1,1, 3],
                  [2, 1,1,1,2, 2],
                  [3, 1,1,2,1, 3],
                  [4, 1,1,2,2, 1],
                  [5, 1,2,1,1, 3],
                  [6, 1,2,1,2, 2],
                  [7, 1,2,2,1, 3],
                  [8, 1,2,2,2, 1],
                  [9, 2,1,1,1, 3],
                  [10, 2,1,1,2, 2],
                  [11, 2,1,2,1, 3],
                  [12, 2,1,2,2, 1],
                  [13, 2,2,1,1, 3],
                  [14, 2,2,1,2, 2],
                  [15, 2,2,2,1, 3],
                  [16, 2,2,2,2, 3],
                  [17, 3,1,1,1, 3],
                  [18, 3,1,1,2, 3],
                  [19, 3,1,2,1, 3],
                  [20, 3,1,2,2, 1],
                  [21, 3,2,1,1, 3],
                  [22, 3,2,1,2, 2],
                  [23, 3,2,2,1, 3],
                  [24, 3,2,2,2, 3]])

# 4 columns in the middle for Features
trainX = data[:, 1:5]
# first 19 rows for training data
trainY = data[0:19, :]
# last 5 rows for test data
testX = data[19:24, :]

# L1: Manhattan Distance
for i in trainX: 
    distance_L1 = 0
    val = abs(ai - bi)
    distance_L1 += val
    print('Manhattan distance is: ', distance_L1)
    
# L2: Euclidean Distance
for i in something: 
    distance_L2 = 0
    val = (ai - bi)**2
    distance_L2 += val
    print('Euclidean distance is: ', math.sqrt(distance_L2))

# L3: Chelyshev Distance
math.max(abs(ai - bi))


def myKNN(trainX, trainY, testX, distance, K):
    """ trainX <- training input features 
        trainY <- training labels
        testX <- test dataset 
        distance determines the distance metric and can be 1, 2, 3 
        (3 for  L_chelyshev} ). Also, K is the KNN parameter. """
    trainX = data[:, 1:5]
    trainY = data[0:19, :]
    testX = data[19:24, :]
    
#     # distance equations
#     L1_manhattan = math.sum((abs(ai - bi)))
#     L2_euclidean = math.sum((ai - bi)**2)
#     L_chelyshev = math.max(abs(ai - bi))

#################### STEP 2 - Decision Tree Toy Example ######################

# from sklearn.tree import DecisionTreeClassifier

# X = [[0, 0], [1, 1], [0, 1], [2, 2]]
# Y = [0, 1, 0, 1]
# clf = DecisionTreeClassifier()
# clf = clf.fit(X, Y)
# clf.predict([[1, 2]])


################# STEP 3 - Decision Tree With Larger Dataset #################

       
       
       
       
     