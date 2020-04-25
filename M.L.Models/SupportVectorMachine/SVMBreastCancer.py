#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 07:18:23 2020

@author: deepu
"""

#************************* SVM on BreastCancer*********************************

#importing libraries
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix 
from sklearn.svm import SVC  #support vector classification
from sklearn.model_selection import GridSearchCV 
from sklearn import svm

#load the dataset
cancer = load_breast_cancer()

# The data set is presented in a dictionary form: 
#print(cancer.keys()) 

#features
df_feat = pd.DataFrame(cancer['data'], columns = cancer['feature_names']) 
  
# cancer column is target 
df_target = pd.DataFrame(cancer['target'], columns =['Cancer']) 
  
#print("Feature Variables: ") 
#print(df_feat.info()) 

#print("Dataframe looks like : ") 
#print(df_feat.head()) 

#np.ravel is reshape the data as contiguous array as reshape(-1)
X_train, X_test, y_train, y_test = train_test_split(df_feat, 
                                                    np.ravel(df_target), 
                                                    test_size = 0.30,
                                                    random_state = 101)

#Train the Support Vector Classifier without Hyper-parameter Tuning
model = SVC()
model.fit(X_train,y_train)

#prediction
predict = model.predict(X_test)

#generate classification report
clsReport = classification_report(y_test, predict)

#onbtain accuracy is 61 % &
#the obtain precision and recall for column '0' is zero it 
#means there is only one class is class '1' 
#This means model neads parameter tuned so apply GridSearchCV.

# defining parameter range 
paramGrid = {'C': [0.1, 1, 10, 100, 1000],  
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
              'kernel': ['rbf']}

grid = GridSearchCV(SVC(), paramGrid, refit = True, verbose = 3)

#it calculate the score for every poosible combination of gridSearch
#the combination whose score is highest it wil return that value of 
# 'C' and 'gamma' from that combination 
grid.fit(X_train,y_train)

# print best parameter after tuning 
gridBestParam = grid.best_params_
  
# print how our model looks after hyper-parameter tuning 
gridBestEstimator = grid.best_estimator_

gridPredictions = grid.predict(X_test) 

# print Grid classification report 
gridClsReport = classification_report(y_test, gridPredictions)
 
#********************implemaintaing SVM*****************************
X = cancer.data[:,:2]
y = np.array(df_target)
y = np.ravel(y)
# create a mesh to plot in
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

h = (x_max / x_min)/100
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
 np.arange(y_min, y_max, h))

#accessing values from the dictionary [two ways to access]
C = gridBestParam.get("C")   #1
gamma = gridBestParam["gamma"] #2
kernel = gridBestParam["kernel"]

svc = svm.SVC(kernel=kernel, C=C,gamma=gamma).fit(X, y)

plt.subplot(1, 1, 1)
Z = svc.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.1)

plt.scatter(X[:, 0], X[:, 1], c=y , cmap=plt.cm.Paired)
plt.xlabel('mean radius')
plt.ylabel('mean texture')
plt.xlim(xx.min(), xx.max())
plt.title('SVC with rbf kernel')
plt.show()

#******************************Concept of GridSearchCV************************
#https://www.geeksforgeeks.org/svm-hyperparameter-tuning-using-gridsearchcv-ml/
#*****************************************************************************