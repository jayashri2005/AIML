from sklearn.linear_model import LogisticRegression 
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_moons #to generate synthetic data
import matplotlib.pyplot as plt
import mglearn as mg
import matplotlib 
import numpy as np
matplotlib.use('Agg')


x,y=make_moons(noise=0.15,random_state=21) #data generate
mmod = LogisticRegression() #model create
mmod.fit(x,y) #model train

plt.figure(figsize=(10, 6)) #plot size
mg.discrete_scatter(x[:,0], x[:,1], y) #scatter plot

w = mmod.coef_[0] #weight
b = mmod.intercept_[0] #bias
x_boundary = np.linspace(x[:,0].min()-1, x[:,0].max()+1, 100) #x axis
y_boundary = -(w[0] * x_boundary + b) / w[1] #y axis
plt.plot(x_boundary, y_boundary, 'k-', linewidth=2, label='Decision Boundary') 

plt.legend()
plt.title('Logistic Regression with Decision Boundary')
plt.savefig('regression2.png')



#using mlp

mmod1=MLPClassifier(max_iter=1000, random_state=42) 
mmod1.fit(x,y)
y_pred_mlp=mmod1.predict(x)

# Create MLP plot with decision boundary
plt.figure(figsize=(10, 6))

# Create a mesh grid for decision boundary
h = 0.02  # step size in mesh
x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# Predict on mesh grid
Z = mmod1.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot decision boundary as contour
plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdYlBu)
mg.discrete_scatter(x[:,0], x[:,1], y_pred_mlp)
plt.legend()
plt.title('MLP Classifier with Decision Boundary')
plt.savefig('regression3.png')
