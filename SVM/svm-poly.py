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

#kernal = poly 
#svm - polynomial 
#huge error -> r2 score fails -> use -> polynomial reg or decision tree or svm as kernel as linear 

mod3 = SVC(kernel='poly',degree=3)
mod3.fit(X_train,y_train)
mg.discrete_scatter(X_train[:,0],X_train[:,1],y_train)
mg.plots.plot_2d_separator(mod3,X_train)
plt.savefig("svm4.png")
