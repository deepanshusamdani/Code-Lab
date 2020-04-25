#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 15:34:48 2020

@author: deepu
"""
#**************************DCT on indianDiabetes data********************

#importing libraries
import pandas as pd
import graphviz
import pydotplus  #convert dot file into png(decision tree)
from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.tree import export_graphviz #convert img into dot file

#Loading Data
colNames = ['pregnant','glucose','bp','skin','insulin',
            'bmi','pedigree','age','label']

data = pd.read_csv('/home/deepu/Desktop/DataSet/indianDiabetes.csv',
                   names=colNames)

#feature selection
X = data[data.columns[:-1]]   #features
y = data[data.columns[-1]]    #target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, 
                                                    random_state=1)

#********Building Decision Tree Model*************

#Create Decision Tree classifer object
clf = DecisionTreeClassifier()

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

#Model Accuracy
Accuracy = metrics.accuracy_score(y_test, y_pred)

dot_data = StringIO()

export_graphviz(clf, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True,
                feature_names = data.columns[:-1],
                class_names=['0','1'])

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('/home/deepu/Desktop/diabetes.png')
decisionTree = Image(graph.create_png())

# Create Decision Tree classifer object
clfEntropy = DecisionTreeClassifier(criterion="entropy", max_depth=3)

# Train Decision Tree Classifer
clfEntropy = clfEntropy.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clfEntropy.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("AccuracyEntropy:",metrics.accuracy_score(y_test, y_pred))