from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mglearn as mg

from sklearn.metrics import accuracy_score

dt=load_iris()
dt.keys()

x=np.linspace(2,8,51)
fx=np.sin(x)
plt.plot(x,fx)
plt.savefig('decision1.png')

np.random.seed(101)
y=fx+np.random.normal(0,0.6,51)
y=np.round(y,2)
plt.scatter(x,y)
plt.savefig('decision2.png')

from sklearn.tree import DecisionTreeRegressor
model=DecisionTreeRegressor()
X=x.reshape(-1,1)
model.fit(X,y)

yp=model.predict(X)
plt.scatter(x,y)
plt.plot(x,yp,':')
plt.savefig('decision3.png')

#train-test split for 1 used to train nd get model and another remains same 
import pandas as pd
x=np.linspace(2,8,51)
fx=np.sin(x)
np.random.seed(101)
y=fx+np.random.normal(0,0.6,51)
y=np.round(y,2)
from sklearn.model_selection import train_test_split

print(train_test_split([10,20,30,40,50,60,70,80,90,100],list('abcdefghij')))
print(train_test_split(X,y,random_state=1))
xtrain,xtest,ytrain,ytest=train_test_split(X,y,random_state=42)
print(xtrain.shape)
print(xtest.shape)

train = np.hstack((xtrain.reshape(-1,1),ytrain.reshape(-1,1)))
nfTrain=pd.DataFrame(train,columns=['x','y'])
test = np.hstack((xtest.reshape(-1,1),ytest.reshape(-1,1)))
nfTest=pd.DataFrame(test,columns=['x','y'])
nfTrain.sort_values(by='x',inplace=True)
nfTest.sort_values(by='x',inplace=True)
print(nfTrain['x'].values)
print(nfTest['x'].values)

modd = DecisionTreeRegressor()
modd.fit(xtrain,ytrain)
yp_train = modd.predict(xtrain)
yp_test = modd.predict(xtest)

from sklearn.metrics import mean_squared_error,r2_score
print(mean_squared_error(nfTrain['y'].values,yp_train))
print(mean_squared_error(nfTest['y'].values,yp_test))
print(r2_score(nfTrain['y'].values,yp_train))
print(r2_score(nfTest['y'].values,yp_test))

plt.scatter(nfTrain['x'],nfTrain['y'])
plt.plot(nfTrain['x'],yp_train,':')
plt.savefig('decision4.png')

plt.scatter(nfTest['x'],nfTest['y'])
plt.plot(nfTest['x'],yp_test,':')
plt.savefig('decision5.png')

plt.figure(figsize=(20,20))
plot_tree(modd, filled=True)
plt.savefig('tree_full.png')

#random forest tree

from sklearn.ensemble import RandomForestRegressor
modrf=RandomForestRegressor(n_estimators=100)
modrf.fit(nfTrain['x'].values.reshape(-1,1),nfTrain['y'].values)
yp_trainrf = modrf.predict(nfTrain['x'].values.reshape(-1,1))
yp_testrf = modrf.predict(nfTest['x'].values.reshape(-1,1))
plt.scatter(nfTrain['x'],nfTrain['y'])
plt.plot(nfTrain['x'],yp_trainrf,':')
plt.savefig('decision6.png')

plt.scatter(nfTest['x'],nfTest['y'])
plt.plot(nfTest['x'],yp_testrf,':')
plt.savefig('decision7.png')

