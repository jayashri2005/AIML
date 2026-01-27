#train-test split for 1 used to train nd get model and another remains same 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.model_selection import train_test_split
import mglearn as mg
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

X =np.linspace(2,8,51)
fx=np.sin(X)
np.random.seed(101)
y=fx+np.random.normal(0,0.6,51)
y=np.round(y,2)
from sklearn.model_selection import train_test_split

print(train_test_split([10,20,30,40,50,60,70,80,90,100],list('abcdefghij')))
print(train_test_split(X,y,random_state=1))
xtrain,xtest,ytrain,ytest=train_test_split(X,y,random_state=41)
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
modd.fit(xtrain.reshape(-1,1),ytrain)
yp_train = modd.predict(xtrain.reshape(-1,1))
yp_test = modd.predict(xtest.reshape(-1,1))

from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_percentage_error
print(mean_squared_error(nfTrain['y'].values,yp_train))
print(mean_squared_error(nfTest['y'].values,yp_test))
print(r2_score(nfTrain['y'].values,yp_train))
print(r2_score(nfTest['y'].values,yp_test))
print(mean_absolute_percentage_error(nfTrain['y'].values,yp_train))
print(mean_absolute_percentage_error(nfTest['y'].values,yp_test))

plt.scatter(nfTrain['x'],nfTrain['y'])
plt.plot(nfTrain['x'],yp_train,':')
plt.savefig('decisiontest1.png')

plt.scatter(nfTest['x'],nfTest['y'])
plt.plot(nfTest['x'],yp_test,':')
plt.savefig('decisiontest2.png')

plt.scatter(nfTrain['x'],nfTrain['y'])
plt.scatter(nfTest['x'],nfTest['y'])
plt.plot(nfTrain['x'],yp_train)

plt.savefig('decisiontest3.png')
print(r2_score(nfTrain['y'].values,yp_train))
print(r2_score(nfTest['y'].values,yp_test))

"""
plt.figure(figsize=(20,20))
plot_tree(modd, filled=True)
plt.savefig('tree_full.png')
""" 
mod1=DecisionTreeRegressor(max_depth=3)
mod1.fit(nfTrain['x'].values.reshape(-1,1),nfTrain['y'].values)
yp_train1 = mod1.predict(nfTrain['x'].values.reshape(-1,1))
yp_test1 = mod1.predict(nfTest['x'].values.reshape(-1,1))

plt.scatter(nfTrain['x'],nfTrain['y'])
plt.scatter(nfTest['x'],nfTest['y'])
plt.plot(nfTrain['x'],yp_train1,':')
plt.savefig('decisiontest4.png')

print(r2_score(nfTrain['y'].values,yp_train1))
print(r2_score(nfTest['y'].values,yp_test1))

#increasing depth 


mod2=DecisionTreeRegressor(max_depth=5)
mod2.fit(nfTrain['x'].values.reshape(-1,1),nfTrain['y'].values)
yp_train2 = mod2.predict(nfTrain['x'].values.reshape(-1,1))
yp_test2 = mod2.predict(nfTest['x'].values.reshape(-1,1))

plt.scatter(nfTrain['x'], nfTrain['y'])
plt.scatter(nfTest['x'], nfTest['y'])
plt.plot(nfTrain['x'], yp_train2)
plt.savefig('decisiontest5.png')
     
print(r2_score(nfTrain['y'].values,yp_train2))
print(r2_score(nfTest['y'].values,yp_test2))
print(mean_squared_error(nfTrain['y'].values,yp_train2))
print(mean_squared_error(nfTest['y'].values,yp_test2))

from sklearn.ensemble import RandomForestRegressor
modrf=RandomForestRegressor(n_estimators=100)
modrf.fit(nfTrain['x'].values.reshape(-1,1),nfTrain['y'].values)
yp_trainrf = modrf.predict(nfTrain['x'].values.reshape(-1,1))
yp_testrf = modrf.predict(nfTest['x'].values.reshape(-1,1))
plt.scatter(nfTrain['x'],nfTrain['y'])
plt.plot(nfTrain['x'],yp_trainrf,':')
plt.savefig('decisiontest6.png')

plt.scatter(nfTest['x'],nfTest['y'])
plt.plot(nfTest['x'],yp_testrf,':')
plt.savefig('decisiontest7.png')

print(r2_score(nfTrain['y'].values,yp_trainrf))
print(r2_score(nfTest['y'].values,yp_testrf))
print(mean_squared_error(nfTrain['y'].values,yp_trainrf))
print(mean_squared_error(nfTest['y'].values,yp_testrf))

#Restricting its depth

randmodel1=RandomForestRegressor(n_estimators=100,max_depth=5)
randmodel1.fit(nfTrain['x'].values.reshape(-1,1),nfTrain['y'].values)
yp_trainrand1 = randmodel1.predict(nfTrain['x'].values.reshape(-1,1))
yp_testrand1 = randmodel1.predict(nfTest['x'].values.reshape(-1,1))

plt.scatter(nfTrain['x'], nfTrain['y'])
plt.scatter(nfTest['x'], nfTest['y'])
plt.plot(nfTrain['x'], yp_trainrand1)
plt.savefig('decisiontest8.png')

print(mean_absolute_percentage_error(nfTrain['y'].values,yp_trainrand1))
print(mean_absolute_percentage_error(nfTest['y'].values,yp_testrand1))
  

