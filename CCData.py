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
origDataset = pd.read_csv('coin_track.csv')
# Drop rows containing intermediate totals
for i in range(0,len(origDataset['Total'])):
    if 'TOTAL' in origDataset['STATE/UT'][i]:
        origDataset.drop(i, inplace=True)
    elif 'TOTAL' in origDataset['CRIME HEAD'][i]:
        origDataset.drop(i, inplace=True)
#remove index gaps
dataset = pd.DataFrame()
dataset = dataset.append(origDataset, ignore_index=True)

dataset2011 = []
for i in range(0,len(dataset['Total'])):
    if dataset['YEAR'][i] == 2011:
        dataset2011.append(dataset.iloc[i])

dataset2011 = pd.DataFrame(dataset2011, index=np.arange(len(dataset2011)))

# Compute the total number of cases per State/UT
catenc = pd.factorize(dataset['STATE/UT'])
sum_states = np.zeros((len(catenc[1]),1))
for i in range(len(catenc[1])):
    for j in range(len(dataset['Total'])):
        if dataset['STATE/UT'][j] == catenc[1][i]:
            sum_states[i] = sum_states[i] + dataset['Total'][j]

# Plot bar chart per State/UT
plt.figure(figsize=(16, 16))
plt.title('Cybercrime cases by State/UT 2008-2013')
plt.xlabel('State/UT')
plt.ylabel('Total CC cases')
plt.xticks(range(len(catenc[1])), catenc[1], rotation=-90)
plt.bar(np.arange(len(sum_states)), sum_states)
#plt.show()
plt.savefig('State_UI_chart.png')

# Compute the total number of cases by type
catenc = pd.factorize(dataset['CRIME HEAD'])
sum_states = np.zeros((len(catenc[1]),1))
for i in range(len(catenc[1])):
    for j in range(len(dataset['Total'])):
        if dataset['CRIME HEAD'][j] == catenc[1][i]:
            sum_states[i] = sum_states[i] + dataset['Total'][j]

# Plot bar chart per State/UT
plt.figure(figsize=(16, 16))
plt.title('Cybercrime cases by type 2008-2013')
plt.xlabel('Crime Type')
plt.ylabel('Total CC cases')
plt.xticks(range(len(catenc[1])), catenc[1], rotation=-90)
plt.bar(np.arange(len(sum_states)), sum_states)
#plt.show()
plt.savefig('Crime_Type_chart.png')

# Compute the total number of cases by age group
age_categories = ['Below 18 Years', 'Between 18-30 Years', 'Between 30-45 Years', 'Between 45-60 Years',
       'Above 60 Years']
sum_ages = np.zeros((len(age_categories),1))
for i in range(len(age_categories)):
    sum_ages[i] = sum(dataset[age_categories[i]])

# Plot pie chart by age
plt.figure(figsize=(10, 10))
plt.title('Cybercrime cases by age 2008-2013')
plt.pie(sum_ages, labels=age_categories)
plt.axis('equal')
#plt.show()
plt.savefig('Crime_Age_chart.png')

# Compute the 2011 total cases along with number of computers
catenc = pd.factorize(dataset2011['STATE/UT'])
sum_states = np.zeros((len(catenc[1]),1))
comp_states = np.zeros((len(catenc[1]),1))
for i in range(len(catenc[1])):
    for k in range(len(compdata2011['State'])):
        if compdata2011['State'][k] == catenc[1][i]:
            comp_states[i] = compdata2011['Computers'][k]
            break
    for j in range(len(dataset2011['Total'])):
        if dataset2011['STATE/UT'][j] == catenc[1][i]:
            sum_states[i] = sum_states[i] + dataset2011['Total'][j]


# Plot bar chart per State/UT
#plt.figure(figsize=(16, 16))
#plt.title('Cybercrime cases by State/UT 2008-2013')
#plt.xlabel('State/UT')
#plt.ylabel('Total CC cases')
#plt.xticks(range(len(catenc[1])), catenc[1], rotation=-90)
#plt.bar(np.arange(len(sum_states)), sum_states)
#plt.bar(np.arange(len(sum_states))+0.5, comp_states)

sum_states = sum_states/max(sum_states)
comp_states = comp_states/max(comp_states)

#ax = plt.subplot(111)
#ax.bar(np.arange(len(sum_states)), sum_states,width=0.2,color='b',align='center')
#ax.bar(np.arange(len(sum_states))+0.2, comp_states,width=0.2,color='r',align='center')
#plt.xticks(range(len(catenc[1])), catenc[1], rotation=-90)
#
#plt.show()

#plt.show()

# Plot bar chart for 2011
plt.figure(figsize=(16, 16))
plt.title('Cybercrime and No. of PCs (normalized) 2011')
plt.xlabel('State/UT')
plt.ylabel('Normalized CC cases & Number of PCs')
plt.xticks(range(len(catenc[1])), catenc[1], rotation=-90)
plt.bar(np.arange(len(sum_states)), sum_states,width=0.2,color='r',align='center')
plt.bar(np.arange(len(sum_states))+0.2, comp_states,width=0.2,color='b',align='center')


plt.savefig('States_Comps_2011_chart.png')


