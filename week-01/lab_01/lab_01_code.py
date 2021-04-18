# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 22:19:29 2021

@author: ANalundasan
For: OMSBA 5067, Lab 1
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
##########################################
## Step 5: 
# P = np.random.normal(0,2,1000)


# plt.hist(P,100)
# plt.show()

# The normal random number generator of numpy has three main parameters: 
    # μ, σ, and N, where the first two ones are the distribution's parameter 
    # and the third variable is the number of samples we want to generate.
    
# P = np.random.normal(0,2,1000)

# generates 1000 random numbers with μ=0 and σ=2.

# step_5a = np.random.normal(-2, 1, 10000)
# plt.hist(step_5a, 100)
# plt.show()

# step_5b = np.random.normal(0, 5, 10000)
# plt.hist(step_5b, 100)
# plt.show()

# step_5c = np.random.normal(5, 0.5, 10000)
# plt.hist(step_5c, 100)
# plt.show()

##########################################
## Step 6

# # Update Feature C to 5 for when Feature A >= 3
# df.loc[df.Feature_A >= 3, "Feature_C"] = 5

# # We can also add new columns to the DataFrame using the "where" function
# df["label"] = np.where(df["Feature_A"] > 2, "big", "small")

# # Or, we can subtract a value from one column
# df["Feature_A"] = df.Feature_A - 5

##########################################
## Step 7
dc = dict()
dc["Label"] = ["big","small","big","small"]
dc["Fixed"] = 1
dc["Variable"] = np.array([2000,5,3000,4])

df_from_dict = pd.DataFrame(dc)

##########################################
## Step 8
# pd.merge(df_from_dict, df, left_on="Label",right_on="label")

##########################################
## Step 9
# Generates three sets of numpy normal random numbers with means of -2, 0, 
# and +2, and with standard deviations equal to 1. Each set should contain 
# 2000 random numbers.

R_neg2 = np.random.normal(-2, 1, 2000)
R_zero = np.random.normal(0, 1, 2000)
R_pos2 = np.random.normal(2, 1, 2000)

# Puts these arrays in a dictionary with these keys: "R_neg2", "R_zero", and "R_pos2".

step_9_b = dict()
step_9_b["R_neg2"] = np.array(R_neg2)
step_9_b["R_zero"] = np.array(R_zero)
step_9_b["R_pos2"] = np.array(R_pos2)

# # Converts the dictionary to a DataFrame, name it as "df"

df = pd.DataFrame(step_9_b)

# # Calculates the statistics of each column (using describe())
df.describe()

# # Plots the histogram of the three sets of numbers on a same graph.
plt.hist([R_neg2, R_zero, R_pos2], 100)
plt.show

all_low = np.where((df["R_neg2"]<=-2) & (df["R_zero"]<=-2) & (df["R_pos2"]<=-2))
all_high = np.where((df["R_neg2"]>=2) & (df["R_zero"]>=2) & (df["R_pos2"]>=2))
print("All Low Count: ",len(all_low[0])," All High Count: ",len(all_high[0]))
