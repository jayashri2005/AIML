from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
    
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

x = np.linspace(2,8,51)
fx = np.sin(x)
np.random.seed(101)
y = fx + np.random.normal(0,0.6,51)
Y = y.round(2)
X = x.reshape(-1,1)
print(X.shape)

X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=42)

model = AdaBoostRegressor(
    estimator=DecisionTreeRegressor(max_depth=1),
    n_estimators=500,
    random_state=42,
    learning_rate=0.05
)

model.fit(X_train, y_train)
yp_train = model.predict(X_train)
y_pred = model.predict(X_test)

print("R2:", r2_score(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))

plt.figure(figsize=(10,6))
plt.scatter(X_train, y_train, label="y_train")
plt.scatter(X_test, y_test, label="y_test")
plt.scatter(X_train, yp_train, label="y_pred")
plt.legend()
plt.savefig('adaboosting.png')

