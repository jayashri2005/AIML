import torch
import torch.nn as nn
import numpy as np

np.random.seed(101)
xx=np.linspace(3,8,11)
yy=xx**2+3+np.random.randn(11)
xx=xx.reshape(-1,1)
#yy=yy.reshape(-1,1)
xx=torch.FloatTensor(xx)
yy=torch.FloatTensor(yy)
mod1=nn.Sequential(nn.Linear(1,1))
print(mod1(xx))


from torch.optim import Adam
from torch.optim import SGD
lossfn1=nn.MSELoss()
sgd=SGD(mod1.parameters(),lr=0.01)
for i in range(100):
    sgd.zero_grad()
    y_pred=mod1(xx)
    loss=lossfn1(y_pred,yy)
    loss.backward()
    sgd.step()
    print(loss.item())