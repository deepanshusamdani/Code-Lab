#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Tue Mar 31 17:49:14 2020

@author: deepu
"""

#*******************Logistic Regression Model on Titanic***********************

#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import Imputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

#read the data set
data = pd.read_csv('/home/deepu/Desktop/DataSet/Titanic/titanicTrain.csv')

#to check where the data is NAN using seaborn
sns.heatmap(data.isnull(),
            xticklabels=True,       #if True: Show X scale name
            yticklabels=False,      #if True: show y scale name  
            cbar=True,              #if True: Show Color bar
            cmap='viridis')         #Bg color

#count-plot of people survided according to male/female
sns.set_style('whitegrid')
plt.show()
sns.countplot(x='Survived', hue='Sex', data=data, palette='RdBu_r')

#no. of people who survived according to their Passenger Class
sns.set_style('whitegrid')
plt.show()
sns.countplot(x='Survived', hue='Pclass', data=data)

#countplot of the people having siblings or spouce
sns.countplot(x='SibSp',data=data)
plt.show()

#distribution plot of the ticket fare
data['Fare'].hist(color='green',bins=40,figsize=(8,4))
plt.show()


#*******Applying Feature Engineering to prepare model*******

#create DataFrame using pandas
newData = pd.DataFrame(data)
newVolume = newData.dropna()

# drop NAN from Age
newSet = newData[newData['Age'].isnull()]
#newSet = newSet.dropna()

Agemean=newVolume['Age'].mean()
print(Agemean)
newData['Age'] = newData.Age.replace(np.NaN, Agemean)

sns.heatmap(newData.isnull(),yticklabels=False,cbar=True,cmap='viridis')
plt.show()

#drop the column "Cabin" which contain NaN data: because not so much of use
newData.drop(['Cabin'],axis=1,inplace=True)
newData.dropna(inplace=True)

#again check is there ant field where NaN is present
sns.heatmap(newData.isnull(),yticklabels=False,cbar=True,cmap='viridis')
plt.show()

#remove the non numeric values and convert those value into numeric if possible 
#NOTE: In this col. there are only two values(M,F), So i use this metod to covert
newData.Sex[newData.Sex == 'male'] = 1
newData.Sex[newData.Sex == 'female'] = 0

#Now Changes in the column Embarked: (their are multiple values so use method:)
#By using One Hot Encoding Method(get_dummies)
embark = pd.get_dummies(newData["Embarked"], prefix='Emb' ,drop_first=True)

#Now Drop the all NoN Numeric Columns
newData.drop(['Embarked','Name','Ticket'],axis=1,inplace=True)

#concat the column "embark" column to newData
newData = pd.concat([newData,embark],axis=1)

#**************Building Logistic Reg Model******************************

X_train, X_test, y_train, y_test = train_test_split(
                        newData.drop(['Survived'],axis=1), 
                        newData['Survived'], 
                        test_size=0.30, 
                        random_state=101)

#create an instance and fit the model 
logmodel = LogisticRegression()
logmodel.fit(X_train, y_train)

#predict , we can also apply logmodel.predict_proba(X_test)
Predictions = logmodel.predict(X_test)

#Model Evaluation {classification report}
target_names = ['Survive 0','Survive 1']
clasReport = classification_report(y_test,
                                   Predictions,
                                   target_names=target_names)

#Confusion matrix 
confMatrix = confusion_matrix(y_test, Predictions)


#********************* Some Other Concept ******************************
#drop the NAN values record 
#dropNANdata = data.dropna()

#Only the data which contain NAN those all rows are return
#nullData=data[data.isnull().any(axis=1)]

#get numeric data from the whole dataset
#numColFromData = nullData._get_numeric_data()

#X=numColFromData.iloc[:, :-1].values
#X=data.iloc[:, :].values

#imputer=Imputer(missing_values='NaN',strategy='mean',axis=0)
#imputer= imputer.fit_transform(X[:,:])
#X[: , :]= imputer.transform(X[:, :])

#********************* Some Ref. Link ********************************
"""                                                                      
%DataSet:
@: https://www.kaggle.com/c/titanic/data

%ConceptPart:
1. Log Reg:
@: https://machinelearningmastery.com/logistic-regression-
   for-machine-learning/

2. Encoding Categorical values:(Different Mathods)
@: https://pbpython.com/categorical-encoding.html

3. classification_report:
@: https://scikit-learn.org/stable/modules/generated/sklearn.metrics
    .classification_report.html
    
4. Confusion matrix:
@: https://www.geeksforgeeks.org/confusion-matrix-machine-learning/

%CodePart:
@: https://medium.com/@anishsingh20/logistic-regression-in-
   python-423c8d32838b
"""
#*************************************************************************