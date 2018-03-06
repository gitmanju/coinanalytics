# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 10:45:15 2017

@author: chetan
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
origDataset = pd.read_csv('coin_track_2.csv')

# Compute the total value of coins per location
catenc = pd.factorize(origDataset['Area'])
sum_coins = np.zeros((len(catenc[1]),1))
for i in range(len(catenc[1])):
    for j in range(len(origDataset['Value'])):
        if origDataset['Area'][j] == catenc[1][i]:
            sum_coins[i] = sum_coins[i] + origDataset['Value'][j]

# Plot bar chart per Area
plt.figure(figsize=(16, 16))
plt.title('Value of coins found')
plt.xlabel('Area')
plt.ylabel('Value')
plt.xticks(range(len(catenc[1])), catenc[1], rotation=-90)
plt.bar(np.arange(len(sum_coins)), sum_coins)
#plt.show()
plt.savefig('SumCoins.png')

# Compute the number of coins per location
catenc = pd.factorize(origDataset['Area'])
num_coins = np.zeros((len(catenc[1]),1))
for i in range(len(catenc[1])):
    for j in range(len(origDataset['Number'])):
        if origDataset['Area'][j] == catenc[1][i]:
            num_coins[i] = num_coins[i] + origDataset['Number'][j]

# Plot bar chart per Area
plt.figure(figsize=(16, 16))
plt.title('Number of coins found')
plt.xlabel('Area')
plt.ylabel('Number')
plt.xticks(range(len(catenc[1])), catenc[1], rotation=-90)
plt.bar(np.arange(len(num_coins)), num_coins)
#plt.show()
plt.savefig('NumberOfCoins.png')

# Plot bar chart per Area
plt.figure(figsize=(16, 16))
plt.title('Number and Value of coins found')
plt.xlabel('Area')
plt.ylabel('Number and Value')
plt.xticks(range(len(catenc[1])), catenc[1], rotation=-90)
plt.bar(np.arange(len(sum_coins)), sum_coins,width=0.2,color='r',align='center')
plt.bar(np.arange(len(num_coins))+0.2, num_coins,width=0.2,color='b',align='center')

plt.savefig('NumAndValue.png')


# Compute the total value of coins per sub-area
catenc = pd.factorize(origDataset['SubArea'])
sum_coins = np.zeros((len(catenc[1]),1))
for i in range(len(catenc[1])):
    for j in range(len(origDataset['Value'])):
        if origDataset['SubArea'][j] == catenc[1][i]:
            sum_coins[i] = sum_coins[i] + origDataset['Value'][j]

# Plot bar chart per SubArea
plt.figure(figsize=(16, 16))
plt.title('Value of coins found')
plt.xlabel('SubArea')
plt.ylabel('Value')
plt.xticks(range(len(catenc[1])), catenc[1], rotation=-90)
plt.bar(np.arange(len(sum_coins)), sum_coins)
#plt.show()
plt.savefig('SumCoinsSubArea.png')

# Compute the number of coins per sub-area
catenc = pd.factorize(origDataset['SubArea'])
num_coins = np.zeros((len(catenc[1]),1))
for i in range(len(catenc[1])):
    for j in range(len(origDataset['Number'])):
        if origDataset['SubArea'][j] == catenc[1][i]:
            num_coins[i] = num_coins[i] + origDataset['Number'][j]

# Plot bar chart per sub-Area
plt.figure(figsize=(16, 16))
plt.title('Number of coins found')
plt.xlabel('SubArea')
plt.ylabel('Number')
plt.xticks(range(len(catenc[1])), catenc[1], rotation=-90)
plt.bar(np.arange(len(num_coins)), num_coins)
#plt.show()
plt.savefig('NumberOfCoinsSubArea.png')

# Plot bar chart per sub-Area
plt.figure(figsize=(16, 16))
plt.title('Number and Value of coins found')
plt.xlabel('SubArea')
plt.ylabel('Number and Value')
plt.xticks(range(len(catenc[1])), catenc[1], rotation=-90)
plt.bar(np.arange(len(sum_coins)), sum_coins,width=0.2,color='r',align='center')
plt.bar(np.arange(len(num_coins))+0.2, num_coins,width=0.2,color='b',align='center')

plt.savefig('NumAndValueSubArea.png')
