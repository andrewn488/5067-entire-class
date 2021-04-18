# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 22:22:43 2021

@author: ANalundasan
For: OMSBA 5067, Lab 2
"""

import numpy as np
import matplotlib.pyplot as plt

#################### STEP 1 - TOY SAMPLES ####################################
## GAUSSIAN NAIVE BAYES ##

# X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
# Y = np.array([1, 1, 1, 2, 2, 2])

# from sklearn.naive_bayes import GaussianNB

# gnb = GaussianNB()

# gnb.fit(X, Y)

# print(gnb.predict([[-0.8, 1]]))

## CATEGORICAL NAIVE BAYES ##

# X = np.random.randint(5, size=(6, 100))

# y = np.array([1, 2, 3, 4, 5, 6])

# from sklearn.naive_bayes import CategoricalNB
# clf = CategoricalNB()
# clf.fit(X, y)
# print(clf.predict(X[2:4]))

#################### STEP 2 - SYNTHETIC GAUSSIAN DATA ########################
## SYNTHETIC GAUSSIAN DATA ##
## synthetic dataset for binary classification and Gaussian distribution

# Plen = 1000
# Nlen = 1000
# P = np.random.normal(5,5,Plen)
# N = np.random.normal(-5,5,Nlen)

# X = np.concatenate((np.copy(P),np.copy(N)))
# Y = np.concatenate((np.full(Plen,+1),np.full(Nlen,-1)))

#################### STEP 3 - TRAIN AND TEST SPLIT ##########################
## Train and test split ##

# TrainPortion = 0.8
# msk = np.random.rand(len(X)) < TrainPortion

# trainX = X[msk]
# trainY = Y[msk]

# testX = X[~msk]
# testY = Y[~msk]

# plt.hist(testX)
# plt.hist(testY)
# plt.show()

## By checking the length of train and test portions, verify that the 
    ## splitting is fine.

#################### STEP 4 - TRAIN THE MODEL ################################
## train the model ##

from sklearn.naive_bayes import GaussianNB

# ValueError regarding the vector shapes
# reshape here 
    
# trainX = X[msk].reshape(-1,1)
# trainY = Y[msk]

# testX = X[~msk].reshape(-1,1)
# testY = Y[~msk]

# gnb = GaussianNB()

# gnb.fit(trainX, trainY)

#################### STEP 5 - TEST AND ERROR MEASUREMENT #####################

# estimatedY = gnb.predict(testX)
# misrate = np.sum(np.abs(testY-estimatedY))/(2*len(testY))
# print(misrate)


#################### STEP 6 - EFFECT OF STANDARD DEVIATION ###################
## change standard deviations up in Step 2

#################### STEP 7 - EFFECT OF TRAIN SIZE  ##########################
## change train size in Step 3
# misclassification rate decreases as train size decreases

#################### STEP 8 - DATASET  #######################################

import numpy as np
from numpy import genfromtxt
from sklearn.naive_bayes import GaussianNB

data = genfromtxt('transfusion.csv', delimiter=',', skip_header=1)

## select data from transfusion file
X = data[:, [0, 1, 2, 3]]
Y = data[:, -1]

# train the data at 80% training, 20% testing
TrainPortion = 0.8
msk = np.random.rand(len(X)) < TrainPortion

# train data
trainX = X[msk]
trainY = Y[msk]

# test data
testX = X[~msk]
testY = Y[~msk]

# train the model
gnb = GaussianNB()
gnb.fit(trainX, trainY)

# solve for misclassification
estimatedY = gnb.predict(testX)
misrate = np.sum(np.abs(testY-estimatedY))/len(testY)
print(misrate)




















