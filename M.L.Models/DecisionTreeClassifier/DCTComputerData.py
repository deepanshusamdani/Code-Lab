#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 19:03:43 2020

@author: deepu
"""

#*************DCT on Computer Data without using modules******************** 

#importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import graphviz
import pydotplus  #convert dot file into png(decision tree)
import itertools
from sklearn.tree import export_graphviz #convert img into dot file
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.externals.six import StringIO  
from IPython.display import Image  
from collections import OrderedDict
from itertools import chain


colName = ["RID","age", "income", "student", "creditRating","buysComputer"]
data = pd.read_csv('/home/deepu/Desktop/DataSet/dataCompOne.csv', names = colName)


#createing DataFrame
newData = pd.DataFrame(data)

#globalParameters
GainParam = []

#Calculation of Output class
outputClass = newData['buysComputer'].value_counts()
outclsSum = sum(outputClass)
A = outputClass
B = outclsSum
sums = 0

for index,val in enumerate(outputClass):
    InfoD = (-(A[index]/B)*np.log2(A[index]/B))
    sums += InfoD

#AgeNew = pd.get_dummies(newData["age"], prefix='age')
#for i, count in enumerate(AgeNew):
#    a = AgeNew[AgeNew.columns[i]]
#    aOne = a.value_counts()[1]

ageValCount = newData['age'].value_counts()
ageValSum = sum(ageValCount)
C = ageValCount
D = ageValSum

#function to join the two columns
def get_dict_from_pd(df, key_col, row_col):
    result = dict()
    for i in set(df[key_col].values):
        is_i = df[key_col] == i
        result[i] = list(df[is_i][row_col].values)
        print(type(result))
    return result

#another metod of below "a" ::
#npAge = np.array(AgeDict)
#b = npAge.tolist()
#bSor = sorted(b.items())

AgeDict = get_dict_from_pd(newData, 'age', 'buysComputer')
dict1 = OrderedDict(sorted(AgeDict.items()))
out1 = OrderedDict(sorted(itertools.islice(AgeDict.items(), 3)))
a = list(out1.items())

#split list into seprate list part:
#res = [[i for i, j in a], [j for i,j in a]]
#resZero  = res[0]
#resFirst = res[1]

#creating Empty list
f  = []
CY = []
CN = []
NL = []
final = []

def InfoAgeCall():
    for i in range(0,len(a)):
        CY.append(a[i][1].count('yes'))
        CN.append(a[i][1].count('no'))
        b = a[i][1]
        f.append(b)
        bLen = len(b)
        NL.append(bLen)
        t = sum(NL)
        if (CN[i] != 0 and CY[i] != 0):
            InfoADAge = NL[i]/D * (-(CY[i]/NL[i] * np.log2(CY[i]/NL[i])) 
                                -(CN[i]/NL[i] * np.log2(CN[i]/NL[i]))) 
            final.append(InfoADAge)
            finalSumAge = sum(final)
    return finalSumAge
            
finalSumAge = InfoAgeCall()

    
# calculation for the column name income:
IncomeDict = get_dict_from_pd(newData, 'income', 'buysComputer')

dict2 = OrderedDict(sorted(IncomeDict.items()))
out2 = OrderedDict(sorted(itertools.islice(IncomeDict.items(), 3)))
list2 = list(out2.items())

f2  = []
CY2 = []
CN2 = []
NL2 = []
final2 = []

def InfoIncomeCall():
    for i in range(0,len(list2)):
        CY2.append(list2[i][1].count('yes'))
        CN2.append(list2[i][1].count('no'))
        b = list2[i][1]
        f2.append(b)
        bLen = len(b)
        NL2.append(bLen)
        t2 = sum(NL2)
        if (CN2[i] != 0 and CY2[i] != 0):
            InfoADInc = NL2[i]/D * (-(CY2[i]/NL2[i] * np.log2(CY2[i]/NL2[i])) 
                                -(CN2[i]/NL2[i] * np.log2(CN2[i]/NL2[i]))) 
            final2.append(InfoADInc)
            finalSumIncome = sum(final2)
    return finalSumIncome

finalSumIncome = InfoIncomeCall()

#calculation for the column name student:
StudentDict = get_dict_from_pd(newData, 'student', 'buysComputer')

dict3 = OrderedDict(sorted(StudentDict.items()))
out3 = OrderedDict(sorted(itertools.islice(StudentDict.items(), 3)))
list3 = list(out3.items())

f3  = []
CY3 = []
CN3 = []
NL3 = []
final3 = []

def InfoStudentCall():
    for i in range(0,len(list3)):
        CY3.append(list3[i][1].count('yes'))
        CN3.append(list3[i][1].count('no'))
        b = list3[i][1]
        f3.append(b)
        bLen = len(b)
        NL3.append(bLen)
        t3 = sum(NL3)
        if (CN3[i] != 0 and CY3[i] != 0):
            InfoADStu = NL3[i]/D * (-(CY3[i]/NL3[i] * np.log2(CY3[i]/NL3[i])) 
                                -(CN3[i]/NL3[i] * np.log2(CN3[i]/NL3[i]))) 
            final3.append(InfoADStu)
            finalSumStudent = sum(final3)
    return finalSumStudent

finalSumStudent = InfoStudentCall()



#calculation for the column name creditRating:
creditDict = get_dict_from_pd(newData, 'creditRating', 'buysComputer')

dict4 = OrderedDict(sorted(creditDict.items()))
out4 = OrderedDict(sorted(itertools.islice(creditDict.items(), 3)))
list4 = list(out4.items())

f4  = []
CY4 = []
CN4 = []
NL4 = []
final4 = []

def InfoCreditCall():
    for i in range(0,len(list4)):
        CY4.append(list4[i][1].count('yes'))
        CN4.append(list4[i][1].count('no'))
        b = list4[i][1]
        f4.append(b)
        bLen = len(b)
        NL4.append(bLen)
        t4 = sum(NL4)
        if (CN4[i] != 0 and CY4[i] != 0):
            InfoADCredit = NL4[i]/D * (-(CY4[i]/NL4[i] * np.log2(CY4[i]/NL4[i])) 
                                -(CN4[i]/NL4[i] * np.log2(CN4[i]/NL4[i]))) 
            final4.append(InfoADCredit)
            finalSumCredit = sum(final4)
    return finalSumCredit

finalSumCredit = InfoCreditCall()

# to append all values of InfoGain in the list 
InfoAgeList = [np.array(finalSumAge).tolist()]
InfoIncList = [np.array(finalSumIncome).tolist()]
InfoStuList = [np.array(finalSumStudent).tolist()]
InfoCreList = [np.array(finalSumCredit).tolist()]

GainParam = list(chain(InfoAgeList,InfoIncList,InfoStuList,InfoCreList))

totalGainList=[]
for i in range(len(GainParam)):
    sub = sums - GainParam[i] 
    totalGainList.append(sub)

#for choosing the best split parameter whose Gain is Highest:
#Gain(X) = Info(D) − Info X (D)
bestSplitParaam = max(totalGainList)

#calculate the split info for age:
li = []
spl =[]
def splitInfoAge():
    aa = newData.groupby('age').count()
    adf = pd.DataFrame(aa)
    kk = np.array(adf) 
    for i in range(0,len(kk)):
        xc = kk[i,-1].tolist()
        li.append(xc)
        tot = sum(li)
        split = (-(li[i]/D * np.log2(li[i]/D)))
        spl.append(split)
        tsum = sum(spl)
    return tsum

tsum = splitInfoAge()


#calculate the split info for income:
li1 = []
spl1 =[]
def splitInfoIncome():
    aa1 = newData.groupby('income').count()
    adf1 = pd.DataFrame(aa1)
    kk1 = np.array(adf1) 
    for i in range(0,len(kk1)):
        xc1 = kk1[i,-1].tolist()
        li1.append(xc1)
        tot1 = sum(li1)
        split1 = (-(li1[i]/D * np.log2(li1[i]/D)))
        spl1.append(split1)
        tsum1 = sum(spl1)
    return tsum1

tsum1 = splitInfoIncome()


#calculate the split info for student:
li2 = []
spl2 =[]
def splitInfoStudent():
    aa2 = newData.groupby('student').count()
    adf2 = pd.DataFrame(aa2)
    kk2 = np.array(adf2) 
    for i in range(0,len(kk2)):
        xc2 = kk2[i,-1].tolist()
        li2.append(xc2)
        tot2 = sum(li2)
        split2 = (-(li2[i]/D * np.log2(li2[i]/D)))
        spl2.append(split2)
        tsum2 = sum(spl2)
    return tsum2

tsum2 = splitInfoStudent()


#calculate the split info for creditRating:
li3 = []
spl3 =[]
def splitInfoCredit():
    aa3 = newData.groupby('student').count()
    adf3 = pd.DataFrame(aa3)
    kk3 = np.array(adf3) 
    for i in range(0,len(kk3)):
        xc3 = kk3[i,-1].tolist()
        li3.append(xc3)
        tot3 = sum(li3)
        split3 = (-(li3[i]/D * np.log2(li3[i]/D)))
        spl3.append(split3)
        tsum3 = sum(spl3)
    return tsum3

tsum3 = splitInfoCredit()


# to append all values of SplitInfor in the list 
SplitAgeList = [np.array(tsum).tolist()]
SplitIncList = [np.array(tsum1).tolist()]
SplitStuList = [np.array(tsum2).tolist()]
SplitCreList = [np.array(tsum3).tolist()]

GainRatioParam=[]
GainRatioParam = list(chain(SplitAgeList,SplitIncList,SplitStuList,SplitCreList))


totalGainRatioList=[]
for i in range(len(GainRatioParam)):
    for j in range(len(totalGainList)):
        if(i == j):
            div = totalGainList[j] / GainRatioParam[i] 
            totalGainRatioList.append(div)

#create copy of the data
dataCopy = data.copy()

#Convert str column to numeric one so that it can fit to train the classifier

#feature age
dataCopy.age[dataCopy.age == 'youth']      = 0
dataCopy.age[dataCopy.age == 'middleaged'] = 1
dataCopy.age[dataCopy.age == 'senior']     = 2

#feature income
dataCopy.income[dataCopy.income == 'low']    = 0
dataCopy.income[dataCopy.income == 'medium'] = 1
dataCopy.income[dataCopy.income == 'high']   = 2

#feature student
dataCopy.student[dataCopy.student == 'no']  = 0
dataCopy.student[dataCopy.student == 'yes'] = 1

#feature creditRating
dataCopy.creditRating[dataCopy.creditRating == 'fair']      = 0
dataCopy.creditRating[dataCopy.creditRating == 'excellent'] = 1

#feature buysComputer
dataCopy.buysComputer[dataCopy.buysComputer == 'no']  = 0
dataCopy.buysComputer[dataCopy.buysComputer == 'yes'] = 1


newDataCopy = dataCopy.values
X = []
y = []
for i in range(len(newDataCopy)):
    xx = newDataCopy[i][:-1]
    X.append(xx)
    yy = newDataCopy[i][-1]
    y.append(yy)

        
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.3, 
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
                feature_names = dataCopy.columns[:-1],
                class_names=dataCopy.columns[-1])

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())

graph.write_png('/home/deepu/Desktop/DCT Tree/Comp.png')

decisionTree = Image(graph.create_png())

