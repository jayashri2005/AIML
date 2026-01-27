from sklearn.datasets import load_iris

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score


dt = load_iris()
print(dt.keys())
X = dt.data   
Y = dt.target
print(Y)

modl = DecisionTreeClassifier(criterion='entropy')
modl.fit(X,Y)
print(X.shape)
print(Y.shape)
Yp = modl.predict(X)
plt.figure(figsize=(10,10))
plot_tree(modl)
plt.savefig('tree.png')


mrn = RandomForestClassifier(n_estimators=100)
mrn.fit(X,Y)

print(accuracy_score(Y, Yp))
