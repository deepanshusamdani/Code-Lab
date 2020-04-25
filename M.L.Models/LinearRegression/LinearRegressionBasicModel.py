#!/usr/bin/python3

#import libraries/modules
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from statistics import mean
from matplotlib import interactive

#create datasets
X=np.array([1,2,3,4,5])
Y=np.array([4,12,28,52,80])

#***********applying Simple linear regression(SLR)************** 
# Equation:  y=mx+c 
# slope:	 m=(y2-y1)//(x2-x1)	
# Intercept: y-y1=m(x-x1)
#**************************************************************

#take the parameters from dataset
Xtrain = X[0]
Ytrain = Y[0]
Xtarget = X[-1]
Ytarget = Y[-1]

#calculate slope (m)
def slope_m():
	m = (Ytarget-Ytrain)/(Xtarget-Xtrain)
	return m
m=slope_m()

#calculate intercept (c)
def intercept_c():
	c = Ytrain-(slope_m()*(Xtrain))
	return int(c)
c = intercept_c()

Xbars=['FARMER','VILLAGE','TOWN','CITY','CITYDOWNTOWN']

#calculate cordinates of y
reg_line=[((m*x)+c) for  x in X ]

#error calculation with SLR
diffOrigActual = Y-reg_line
sumSquareError = sum(diffOrigActual**2)

#plot the graph
plt.scatter(X,Y)
plt.plot(X,Y,color='red',linestyle='dashed', marker='o',  markerfacecolor='blue', markersize=10)
plt.plot(X,reg_line,color='green')
plt.title('Simple Lineare Regression')
plt.xlabel('VENDORS',horizontalalignment='center')
plt.ylabel("Product's cost acc to VENDORS",horizontalalignment='center')
plt.xticks(X,Xbars)
#plt.figure(1)
interactive(True)
plt.show()

#alternate method to calculate reg_line(y)
# def cordinates_Y():
# 	for i in range(len(X)):
# 		y = slope_m()*X + intercept_c()
# 		#round off 
# 		y = np.round(y)
# 		#type conversion (float->int)
# 		y = y.astype(int) 
# 	return y
#y = cordinates_()


#*********** applying Least Square Regression(lsr)********************
# Equation:  y = mx+c 
# slope:	 m = SumOfAll((X-Xmean)*(Y-Ymean))/(SumOf((X-Xmean)**2))
# Intercept: C = Ymean - m*Xmean
#**********************************************************************

Xmean= sum(X)//len(X)
Ymean = sum(Y)//len(Y)

#calculate slope
def slope_lsrM():
     lsrM=sum((X-Xmean)*(Y-Ymean))/(sum((X-Xmean)**2)) 
     return lsrM
lsrM = slope_lsrM()

#calculate (X-Xmean)
xcalc = [(x-Xmean) for x in X]

#calculate (Y-Ymean)
ycalc = [(y-Ymean) for y in Y]

#Numerator => SumOfAll((X-Xmean)*(Y-Ymean))
sumOfAll = sum((np.array(xcalc))*(np.array(ycalc)))

#denominator => sumOf(x-Xmean)**2
total = 0
dX = [(total+t) for t in ( i**2 for i in xcalc)]
sumdX= sum(dX)

#caluculate intercept
def  intercept_lsrC():
    lsrC = Ymean - (lsrM)*(Xmean)
    return lsrC 
lsrC = intercept_lsrC()

#calculate reg_line(y) values for X
reg_line=[((lsrM*x)+lsrC) for  x in X ]

#error calculation with SLR
diffOrigActualLSR = Y-reg_line
sumSquareErrorLSR = sum(diffOrigActualLSR**2)

if(sumSquareError > sumSquareErrorLSR):
	print("Error Rate is high %.2f times" %(sumSquareError-sumSquareErrorLSR))
	print("Need to apply Least Square Regression or may need to apply some other model of M.L.")

#plot the graph
plt.scatter(X,Y)
#normal X, Y line
plt.plot(X,Y,color='blue',linestyle='dashed', marker='o',  markerfacecolor='yellow', markersize=10)
#Actual Value
plt.plot(X,reg_line,color='purple')
plt.title('Least Square Regression')
plt.xlabel('LSR VENDORS',horizontalalignment='center')
plt.ylabel("LSR Product's cost acc to VENDORS",horizontalalignment='center')
#plt.xticks(X,Xbars)
#plt.figure(2)
interactive(False)
plt.show()

#*********************************************************************************
#Reference for concept:															 *
#https://towardsdatascience.com/mathematics-									 *
#for-machine-learning-linear-regression-least-square-regression-de09cf53757c     *
#*********************************************************************************