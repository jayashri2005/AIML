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

import mglearn as mg 
mg.discrete_scatter(X_train[:,0],X_train[:,1],y_train)
plt.savefig("svm1.png")

#svm - linear

svm=SVC(kernel='linear',C=0.001)
svm.fit(X_train,y_train)
mg.discrete_scatter(X_train[:,0],X_train[:,1],y_train)
mg.plots.plot_2d_separator(svm,X_train)
plt.savefig("svm2.png")

mod2=SVC(kernel='linear',C=1000)
mod2.fit(X_train,y_train)
mg.discrete_scatter(X_train[:,0],X_train[:,1],y_train)
mg.plots.plot_2d_separator(mod2,X_train)
plt.savefig("svm3.png")
