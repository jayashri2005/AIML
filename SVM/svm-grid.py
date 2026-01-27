from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from sklearn.datasets import make_blobs
from sklearn.svm import SVC 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

X,y = make_blobs(n_samples=100, centers=2, random_state=0,cluster_std=1.0)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

from sklearn.model_selection import StratifiedKFold,cross_val_score
skf=StratifiedKFold(n_splits=5, shuffle=True,random_state=42)
scores = cross_val_score(SVC(kernel="rbf"), X, y, cv=skf,scoring="accuracy")
print(scores)
print(scores.mean())

#gridsearchcv

param_grid ={
    "C": [0.1, 1, 10, 100],
    "gamma": [1, 0.1, 0.01],
    "kernel": ["rbf"]

}
svm =SVC()
grid = GridSearchCV(svm, param_grid, cv=5, scoring="accuracy",n_jobs=-1)
grid.fit(X,y)
print("Best parameters:",grid.best_params_)
plt.savefig("grid_results.png")
