import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


x=np.linspace(0,10,100)
np.random.seed(101)
y=2*x+3+np.random.normal(0,1,100)

y=y.round(2)

plt.scatter(x,y)
plt.savefig('outliers1.png')
plt.clf()

y[-1],y[-2]=99.28,98.28


plt.scatter(x,y)
plt.savefig('outliers2.png')
plt.clf()

X=sm.add_constant(x)
model=sm.OLS(y,X).fit()
influence=model.get_influence()
cooks_d=influence.cooks_distance[0]
print(cooks_d)

model.predict(X)
plt.plot(x,model.predict(X),color='red')
plt.scatter(x,y)
plt.savefig('outliers3.png')

#hubber regressor
from sklearn.linear_model import HuberRegressor
huber=HuberRegressor(epsilon=1.35)  # epsilon controls the transition point (δ)
huber.fit(X,y)
plt.plot(x,huber.predict(X),color='black', label='Huber')
plt.scatter(x,y)
plt.savefig('outliers4.png')

#lasso
from sklearn.linear_model import Lasso
lasso=Lasso(alpha=10.0)  # alpha 
lasso.fit(X,y)
plt.plot(x,lasso.predict(X),color='blue', label='Lasso')
plt.scatter(x,y)
plt.legend()
plt.savefig('outliers5.png')
