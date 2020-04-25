#!/usr/bin/python3


#Basic model of Linear Regresssion apply on Random data

#importing libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

#create a dataset using random function
a = np.random.seed(0)
x = np.random.rand(50,1)
y = 2 + 3*x + np.random.rand(50,1)

# Model initialization
regression_model = LinearRegression()

# Fit the data(train the model)
aR = regression_model.fit(x, y)

#Predict using model
y_predicted = regression_model.predict(x)

# model evaluation
rmse = mean_squared_error(y, y_predicted)
r2 = r2_score(y, y_predicted)

#Showing the values
print('Slope: ' ,regression_model.coef_)
print('Intercept: ', regression_model.intercept_)
print('Root mean squared error: ', rmse)
print('R2 score: ', r2)

#Scatter plot
plt.scatter(x,y,s=10,color='green')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

