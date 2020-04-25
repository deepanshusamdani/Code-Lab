#!/usr/bin/python3

#********************Linear Reg On Iris DataSet*************************

#import libraries/modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns
import warnings

# Seaborn default configuration
sns.set_style("darkgrid")

# set the custom size for my graphs
sns.set(rc={'figure.figsize':(8.7,6.27)})

# filter all warnings
warnings.filterwarnings('ignore') 

# set max column to 999 for displaying in pandas
pd.options.display.max_columns=999 

#read the dataset
data = pd.read_csv('/home/deepu/Desktop/DataSet/iris.csv')

#print(data.head)
#print(data.info)
print("describe",data.describe())

rows, col = data.shape
print("Rows : %s, column : %s" % (rows, col))

#Data Visualization using seaborn
snsdata = data.drop(['Id'], axis=1)

g = sns.pairplot(snsdata
					,hue='Species'
					,markers='+'
					,diag_kind='hist'
					#,x_vars=["SepalWidthCm", "SepalLengthCm"]
					#,y_vars=["PetalWidthCm", "PetalLengthCm"]
				)
#palette = autumn, muted
g = g.map_upper(plt.scatter)
g = g.map_lower(sns.kdeplot)
g = sns.pairplot(snsdata, kind="reg")
plt.show()
sns.violinplot(x='SepalLengthCm', y='Species', data=data, inner='stick', palette='autumn')
plt.show()
sns.violinplot(x='SepalWidthCm', y='Species', data=data, inner='stick', palette='autumn')
plt.show()
sns.violinplot(x='PetalLengthCm', y='Species', data=data, inner='stick', palette='autumn')
plt.show()
sns.violinplot(x='PetalWidthCm', y='Species', data=data, inner='stick', palette='autumn')
plt.show()

#****************Apply Multiple Linear Regression Model******************

mapping = {
    		'Iris-setosa' : 1,
    		'Iris-versicolor' : 2,
    		'Iris-virginica' : 3
		  }

# Input Feature Values
X = data.drop(['Id', 'Species'], axis=1).values 
# Output values
y = data.Species.replace(mapping).values.reshape(rows,1)
# Adding one more column for bias
X = np.hstack(((np.ones((rows,1))), X))
# Setting values of theta randomly
theta = np.random.randn(1,5) 
print("Theta : %s" % (theta))

iteration = 10000

#this is actually alpha.
learning_rate = 0.003
ax = plt.subplot(111)

plt.xlabel("Dataset size", color="Green")
plt.ylabel("Iris Flower (1-3)", color="Green")
plt.title("Iris Flower (Iris-setosa = 1, Iris-versicolor = 2, Iris-virginica = 3)")

ax.legend()
plt.show()

# 1 x 10000 maxtix
J = np.zeros(iteration) 
print("length of J :",len(J))

# Let's train our model to compute values of theta
for i in range(iteration):
    J[i] = (1/(2 * rows) * np.sum((np.dot(X, theta.T) - y) ** 2 ))
    theta -= ((learning_rate/rows) * np.dot((np.dot(X, theta.T) - y).reshape(1,rows), X))

prediction = np.round(np.dot(X, theta.T))

ax = plt.subplot(111)
ax.plot(np.arange(iteration), J)
ax.set_ylim([0,0.15])
plt.ylabel("Cost Values", color="Green")
plt.xlabel("No. of Iterations", color="Green")
plt.title("Mean Squared Error vs Iterations")
plt.show()

ax = sns.lineplot(x=np.arange(iteration), y=J)
plt.show()

ax = plt.subplot(111)
ax.plot(np.arange(1, 151, 1), y, label='Orignal value', color='red')
ax.scatter(np.arange(1, 151, 1), prediction, label='Predicted Value')
plt.xlabel("Dataset size", color="Green")
plt.ylabel("Iris Flower (1-3)", color="Green")
plt.title("Iris Flower (Iris-setosa = 1, Iris-versicolor = 2, Iris-virginica = 3)")

ax.legend()
plt.show()

accuracy = (sum(prediction == y)/float(len(y)) * 100)[0]
print("The model predicted values of Iris dataset with an overall accuracy of %s" % (accuracy))



#**********************Reference Link for Concept**********************************************
#https://seaborn.pydata.org/generated/seaborn.pairplot.html    								  *	
#https://www.kaggle.com/amarpandey/implementing-linear-regression-on-iris-dataset/notebook    *
#**********************************************************************************************