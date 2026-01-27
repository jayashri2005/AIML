from sklearn.datasets import load_iris

dt=load_iris()
print(dt.keys())

x=dt.data
y=dt.target 

from sklearn.tree import DecisionTreeClassifier

mod=DecisionTreeClassifier()
mod.fit(x,y)
print(mod.feature_importances_)
X = x[:, 2:]

import mglearn as mg 
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

mg.discrete_scatter(x1=X[:, 0], x2=X[:, 1], y=y)
plt.legend()
plt.savefig("dtree.png")

mod1=DecisionTreeClassifier()
mod1.fit(X,y)
mg.discrete_scatter(x1=X[:, 0], x2=X[:, 1], y=y)
mg.plots.plot_2d_separator(mod1,X)
plt.legend()
plt.savefig("dtree1.png")

mod2=DecisionTreeClassifier(max_depth=2)
mod2.fit(X,y)
mg.discrete_scatter(x1=X[:, 0], x2=X[:, 1], y=y)
mg.plots.plot_2d_separator(mod2,X)
plt.legend()
plt.savefig("dtree2.png")