from sklearn.datasets import make_circles
from sklearn.svm import SVC 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import mglearn as mg 

#kernel =RBF

X,y = make_circles(n_samples=100, factor=0.3, noise=0.15,random_state=101)

mg.discrete_scatter(X[:,0],X[:,1],y)
plt.savefig("svm5.png")

mod4 = SVC(kernel='rbf',gamma=100)
mod4.fit(X,y)
mg.discrete_scatter(X[:,0],X[:,1],y)
mg.plots.plot_2d_separator(mod4,X)
plt.savefig("svm6.png")
