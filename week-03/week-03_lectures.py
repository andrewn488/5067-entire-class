# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 19:14:05 2021

@author: ANalundasan
For: OMSBA 5067, Week 03 Notes
"""
import math
from math import log
from math import log2

############################### Andrew newb solution ##########################
# px_1 = 0.5
# px_2 = 0.25
# px_3 = 0.125
# px_4 = 0.0625
# px_5 = 0.03125
# px_6 = 0.015625
# px_7 = 0.0078125
# px_8 = 0.0078125

# p_y = 0.125

# entropy1 = p_y * log2(p_y) - px_1 * log2(px_1)
# print('Entropy 1 = ', entropy1)

# entropy2 = p_y * log2(p_y) - px_2 * log2(px_2)
# print('Entropy 2 = ', entropy2)

# entropy3 = p_y * log2(p_y) - px_3 * log2(px_3)
# print('Entropy 3 = ', entropy3)

# entropy4 = p_y * log2(p_y) - px_4 * log2(px_4)
# print('Entropy 4 = ', entropy4)

# entropy5 = p_y * log2(p_y) - px_5 * log2(px_5)
# print('Entropy 5 = ', entropy5)

# entropy6 = p_y * log2(p_y) - px_6 * log2(px_6)
# print('Entropy 6 = ', entropy6)

# entropy7 = p_y * log2(p_y) - px_7 * log2(px_7)
# print('Entropy 7 = ', entropy7)

# entropy8 = p_y * log2(p_y) - px_8 * log2(px_8)
# print('Entropy 8 = ', entropy8)

# entropy_y = -(p_y) * log2(p_y) 
# print('H(Y) = ', entropy_y)
# px = 0.25
# entropy = -px(math.log2(px))
# print('Entropy is: ', entropy)

############################### For Loop Solution ############################

px = [0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125, 0.0078125]
entropy = 0
for p in px: 
    val = p * (math.log2(p))
    print(val)
    entropy += val
print('H(X) is: ', -1 * entropy)

py = [0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125]
entropy = 0
for p in py: 
    val = p * (math.log2(p))
    print(val)
    entropy += val
print('H(Y) is: ', -1 * entropy)
    

# ax = 4
# bx = 3
# ay = 4
# by = 3
# distance = math.sqrt((ax - bx)**2 + (ay - by)**2)
# print('distance is: ', distance)

X = [1, 1.2, 1.5, 2, 2.6, 3, 3.4, 4, 4.2]
Y = [0.5, 0.4, 0.25, 0, -0.3, -0.5, -0.7, -1, -1.1]
split = []
for i in range(len(X)-1):
    split.append((X[i] + X[i+1]) / 2)
print("Split -- ", split)

LC = []
for i in range(len(Y)-1):
    sumY = 0
    for j in range(i+1):
        sumY += Y[j]
    LC.append(sumY / (i+1))
print("\nLC -- ", LC)

LR = []
for i in range(len(LC)):
    LR_value = 0
    for j in range(i+1):
        LR_value += (Y[j]-LC[i]) ** 2 / (i+1)
    LR.append(LR_value)
print("\nLR -- ", LR)

RC = []  
for i in range(1, len(Y)):
    sumY = 0
    for j in range(i, len(Y)):
        sumY += Y[j]
    RC.append(sumY / (len(Y)-i))
print("\nRC -- ", RC)

RR = []
for i in range(len(RC)):
    RR_value = 0
    for j in range(i+1, len(Y)):
        RR_value += (Y[j]-RC[i]) ** 2 / (len(Y)-i-1)
    RR.append(RR_value)
print("\nRR -- ", RR)

TR = [LR[i] + RR[i] for i in range(len(LR))]
print("\nTR -- ", TR)




# x = [1, 2, 5]
# y = [-1, 3, 2]
