#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 07:49:01 2020

@author: deepu
"""

#******************Support Vector Machine Iris**********************

#importing libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets

# import some data to play with
iris = datasets.load_iris()

# we only take the first two features. We could avoid this ugly slicing by using a two-dim dataset
X = iris.data[:, :2] 
y = iris.target

# SVM regularization parameter
C = 1.0 
svc = svm.SVC(kernel='linear', C=1,gamma=0.1).fit(X, y)

# create a mesh to plot in
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

h = (x_max / x_min)/100
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

plt.subplot(1, 1, 1)
Z = svc.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.xlim(xx.min(), xx.max())
plt.title('SVC with linear kernel')
plt.show()

#**********************Reference link******************************************
#https://www.analyticsvidhya.com/blog/2017/09/understaing-support-vector-machine-example-code/
#https://scikit-learn.org/stable/auto_examples/svm/plot_rbf_parameters.html
#https://www.geeksforgeeks.org/svm-hyperparameter-tuning-using-gridsearchcv-ml/
#******************************************************************************
