import seaborn as sns
import statsmodels.api as sm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_csv(r"C:\Emphasis\FirstProject\ML\pipeline\tested.csv")

print("Columns in dataset:", df.columns.tolist())

# Fix column names and select correct columns
dfn = df[['Pclass','Sex','Age','Fare','Survived']].dropna()
print(dfn)

# Create male dummy variable
dfn['male']=pd.get_dummies(dfn['Sex'], drop_first=True).astype(int)
print(dfn)

# Fix the .values() issue and use .values
x=dfn[['Pclass','male','Age','Fare']].values
y=dfn['Survived'].values

# Add constant
X=sm.add_constant(x)
print(X)

# Fix the sm.sm.OLS issue - should be sm.OLS #ols=linear reg to predict error
model=sm.OLS(y,X)
results=model.fit()
print(results.summary())

mod = sm.GLM(y,X)
res=mod.fit()
print(res.summary())

model=sm.Logit(y,X)
res=model.fit()
print(res.summary())

from sklearn.preprocessing import RobustScaler

scaler=RobustScaler()
x_scaled=scaler.fit_transform(x)
print(x_scaled)
