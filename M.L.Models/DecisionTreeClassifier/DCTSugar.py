#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 14:30:36 2020

@author: deepu
"""

#****************************DCT on Sugar Dataset*****************************

#importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import pydotplus  #convert dot file into png(decision tree)
import graphviz
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.tree import export_graphviz #convert img into dot file
from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import DecisionTreeRegressor

#read the dataset
#col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 
             #'bmi', 'pedigree', 'age', 'label']
data = pd.read_csv("/home/deepu/Desktop/DataSet/SugarTest.csv")

newData = pd.DataFrame(data)
newData = newData.drop(['SkinThickness'],axis=1)

X = newData[newData.columns[:-1]]
y = newData[newData.columns[-1]]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, 
                                                    random_state=1)

# Create Decision Tree classifier object
clf = DecisionTreeClassifier()

# Train Decision Tree Classifier
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

#mode Accuracy
Accuracy = metrics.accuracy_score(y_test, y_pred)


dot_data = StringIO()

c = export_graphviz(clf,
                out_file=dot_data,  
                filled=True, 
                rounded=True,
                special_characters=True,
                feature_names = newData.columns[:-1],
                class_names=newData.columns[-1])

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())

graph.write_png('/home/deepu/Desktop/Sugar.png')

decisionTree = Image(graph.create_png())

#apply regression 
rng = np.random.RandomState(1)
Xx = np.sort(5 * rng.rand(80, 1), axis=0)
Yy = np.sin(Xx).ravel()
Yy[::5] += 3 * (0.5 - rng.rand(16))

#Xx = np.sort(newData[newData.columns[1]])
#Yy = np.sin(Xx)
# Fit regression model
regr_1 = DecisionTreeRegressor(max_depth=2)
regr_2 = DecisionTreeRegressor(max_depth=5)
regr_1.fit(Xx, Yy)
regr_2.fit(Xx, Yy)

# Predict
X_testi = np.arange(0, 5, 0.01)[:, np.newaxis]
y_1 = regr_1.predict(X_testi)
y_2 = regr_2.predict(X_testi)

# Plot the results
plt.figure()
plt.scatter(Xx, Yy, s=20, edgecolor="black", c="darkorange", label="data")

plt.plot(X_testi, y_1, color="cornflowerblue",label="max_depth=2", linewidth=2)

plt.plot(X_testi, y_2, color="yellowgreen", label="max_depth=5", linewidth=2)

plt.xlabel("data")
plt.ylabel("target")
plt.title("Decision Tree Regression")
plt.legend()
plt.show() 
