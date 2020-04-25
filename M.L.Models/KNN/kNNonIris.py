#!/usr/bin/env python3

# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 07:56:43 2020

@author: deepu
"""

#*******************************KNN on IrisDataset***************************

#importing libraries
#from random import seed
#from random import randrange
#from csv import reader
#from sklearn.datasets import load_iris
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from math import sqrt
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.model_selection import train_test_split 
from sklearn import metrics


#read the data
dataset = pd.read_csv("/home/deepu/Desktop/DataSet/irisOut.csv")    
data = dataset.values

X = []
y = []
for i in range(len(data)):
    xx = data[i][:-1]
    X.append(xx)
    yy = data[i][-1]
    y.append(yy)

# define model parameter i.e. value of "k"
num_neighbors = 5

# calculate the Euclidean distance between two vectors
def euclideanDistance(row1, row2):
    distance = 0.0
    for i in range(len(row1)-1):
        distance += (row1[i] - row2[i])**2
    return sqrt(distance)

# Locate the most similar neighbors
def getNeighbors(train, test_row, num_neighbors):
    distances = list()
    for train_row in train:
        dist = euclideanDistance(test_row, train_row)
        distances.append((train_row, dist))        
    distances.sort(key=lambda tup: tup[1])
    neighbors = list()
    for i in range(num_neighbors):
        neighbors.append(distances[i][0])
    return neighbors

# Make a classification prediction with neighbors
def predictClassification(train, test_row, num_neighbors):
    neighbors = getNeighbors(train, test_row, num_neighbors)
    output_values = [row[-1] for row in neighbors]
    prediction = max(set(output_values), key=output_values.count)
    return prediction


# Split into training and test set 
X_train, X_test, y_train, y_test = train_test_split( X, y, 
                                                    test_size = 0.2, 
                                                    random_state=42)

#creating Object of knn
knnObj = KNeighborsClassifier(n_neighbors=num_neighbors) 
  
knn = knnObj.fit(X_train, y_train) 

y_pred = knn.predict(X_test)


# define a new record for which predit the output class
row = [5.7,2.9,4.2,1.3]

#calculate the accuracy of the model
Accuracy = metrics.accuracy_score(y_test, y_pred)
print("Accuracy: ",  Accuracy)

#predict the value
label = predictClassification(data, row, num_neighbors)
print('Data=%s, Predicted: %s' % (row, label))


#********************************* Reference Link For Concept *****************************************************
#K-NN: geeksforgeeks.org/k-nearest-neighbor-algorithm-in-python/                                                  *
#k++: https://www.geeksforgeeks.org/ml-k-means-algorithm/?ref=rp                                                  *  
#MLmastery: https://machinelearningmastery.com/tutorial-to-implement-k-nearest-neighbors-in-python-from-scratch/  *
#******************************************************************************************************************


#row0 = dataset[0]
#for row in dataset:
#	distance = euclidean_distance(row0, row)
#	print("D: ",distance)

#neighbors = get_neighbors(dataset, dataset[0], 3)
#for neighbor in neighbors:
#	print("neigh:",neighbor)

#prediction = predict_classification(dataset, dataset[0], 3)
#print('Expected %d, Got %d.' % (dataset[0][-1], prediction))


#method to cnovert the datset column into another form 
"""
# Load a CSV file
def load_csv(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset
 
# Convert string column to float
def str_column_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())
 
# Convert string column to integer
def str_column_to_int(dataset, column):
    class_values = [row[column] for row in dataset]
    unique = set(class_values)
    lookup = dict()
    for i, value in enumerate(unique):
        lookup[value] = i
        print('[%s] => %d' % (value, i))
    for row in dataset:
        row[column] = lookup[row[column]]
    return lookup

# Make a prediction with KNN on Iris Dataset
#filename = '/home/deepu/Desktop/DataSet/newiris.csv'
#dataset = load_csv(filename)
#for i in range(len(dataset[0])-1):
#   str_column_to_float(dataset, i)
# convert class column to integers
#str_column_to_int(dataset, len(dataset[0])-1)

# predict the label
#data = load_iris()
#dataset = data.data

"""

