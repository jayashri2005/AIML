import torch
import torch.nn as nn 
from torch.optim import Adam,SGD

from torchvision import models,transforms,datasets

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5),(0.5))
])
train_dataset = datasets.MNIST(root='./data',train=True,transform=transform,download=True)
print(train_dataset[0][0].shape)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.imshow(train_dataset[0][0].squeeze(), cmap='viridis')
plt.savefig('sample.png')

img = train_dataset.data[:100]
lbs = train_dataset.targets[:100]
import cv2
import numpy as np

ls=[]
for k in range(100):
    img_np=img[k].numpy()
    color = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
    rsz=cv2.resize(color, (224, 224))
    ls.append(rsz)
#print(ls)
ar = np.array(ls)
print(ar.shape)

x=ar.reshape(100,3,224,224)
print(x.shape)

x=torch.FloatTensor(x)
y=torch.tensor(lbs)
print(x.shape)
print(y.shape)

mod = models.vgg16(pretrained=True)
print(mod)

mod.classifier[6]=nn.Linear(4096, 10)
print(mod)

for param in mod.parameters():
    param.requires_grad = False
    
for p in mod.classifier[6].parameters():
    p.requires_grad = True

opt = Adam(mod.classifier[6].parameters(), lr=0.001)
lossfn = nn.CrossEntropyLoss()

for epoch in range(3):
    opt.zero_grad()
    pred = mod(x)
    loss = lossfn(pred, y)
    loss.backward()
    opt.step()
    print(loss.item())
print(x[0].shape)
test = x[0].reshape(1,3,224,224)
sfx=torch.softmax(mod(test),dim=1)
print(torch.argmax(sfx,dim=1))
