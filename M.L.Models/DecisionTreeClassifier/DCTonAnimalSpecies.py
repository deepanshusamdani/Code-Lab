#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 22:38:13 2020

@author: deepu
"""
#************************************DCT on AnimalsSpecies********************************

#importLibraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pydotplus  #convert dot file into png(decision tree)
import graphviz
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.tree import export_graphviz #convert img into dot file
from sklearn.externals.six import StringIO  
from IPython.display import Image  

#read the dataset
columnName = ["toothed","hair","breathes","legs","species"]
data = pd.read_csv("/home/deepu/Desktop/DataSet/AnimalsSpecies.csv",
                   names = columnName)

X = data[data.columns[:-1]]
y = data[data.columns[-1]]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, 
                                                    random_state=1)


#********Building Decision Tree Model*************

#Create Decision Tree classifer object
clf = DecisionTreeClassifier()

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

Accuracy = metrics.accuracy_score(y_test, y_pred)

dot_data = StringIO()

export_graphviz(clf,
                out_file=dot_data,  
                filled=True, 
                rounded=True,
                special_characters=True,
                feature_names = data.columns[:-1],
                class_names=['0','1'])

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())

graph.write_png('/home/deepu/Desktop/Species.png')

decisionTree = Image(graph.create_png())


# Create Decision Tree classifer object
clfEntropy = DecisionTreeClassifier(criterion="entropy", max_depth=3)

# Train Decision Tree Classifer
clfEntropy = clfEntropy.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clfEntropy.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("AccuracyEntropy:",metrics.accuracy_score(y_test, y_pred))

dot_datas = StringIO()

export_graphviz(clf,
                out_file=dot_datas,  
                filled=True, 
                rounded=True,
                special_characters=True,
                feature_names = data.columns[:-1],
                class_names=['0','1'])

graphs = pydotplus.graph_from_dot_data(dot_datas.getvalue())  

graphs.write_png('/home/deepu/Desktop/SpeciesEntropys.png')

decisionTrees = Image(graphs.create_png())