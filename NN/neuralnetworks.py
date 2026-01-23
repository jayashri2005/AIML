from sklearn.neural_network import MLPRegressor
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
import mglearn as mg
import matplotlib
matplotlib.use('Agg')

X,y = make_moons(noise=0.15,random_state=21)
X,y = X[:,0],X[:,1]
X=X.reshape(-1,1)

model = MLPRegressor(hidden_layer_sizes=(10,10,4))
model.fit(X, y)

mg.discrete_scatter(X,y)
y_pred = model.predict(X)
plt.scatter(X, y)
plt.scatter(X, y_pred)
plt.savefig("nn1.png")


#classification

from sklearn.neural_network import MLPClassifier

X,y = make_moons(noise=0.15,random_state=21)
model = MLPClassifier(hidden_layer_sizes=(50,50,20), max_iter=1000)
model.fit(X, y)

# Create new figure for classification
plt.figure()
mg.discrete_scatter(X[:,0],X[:,1],y)
mg.plots.plot_2d_separator(model, X)
y_pred = model.predict(X)
plt.savefig("nn2.png")


#pytorch

import torch
from torch import nn
import torch.optim as optim
from sklearn.datasets import load_iris

model = nn.Sequential(
    nn.Linear(4,10),
    nn.ReLU(),
    nn.Linear(10,3)
)
     
dt = load_iris()
X,y = dt.data,dt.target
X = torch.tensor(X,dtype=torch.float32)
y = torch.tensor(y,dtype=torch.long)

lossfn = nn.CrossEntropyLoss() #internal use softmax 
optimizer = optim.Adam(model.parameters(),lr=0.01)

for _ in range(200):
    optimizer.zero_grad()
    yp=model(X)
    loss=lossfn(yp,y)
    loss.backward()
    optimizer.step()
    print(loss.item())
yyp=model(X)
sfx=torch.softmax(yyp,axis=1)
yp=torch.argmax(sfx,axis=1)
print(yp)