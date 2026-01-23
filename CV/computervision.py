from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
import mglearn as mg
import matplotlib
matplotlib.use('Agg')
import torch
import torch.nn as nn
import torch.optim as optim

dt=load_digits(n_class=3)
dt.images[1]
y=dt.target
images=dt.images
print(y[-2])
print(images[0].shape)
images[0].reshape(1,64)
print(images[0].flatten())
print(images.shape)
x=images.reshape(-1,64)
print(x.shape)

model = nn.Sequential(
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 3)
)
X = torch.tensor(x,dtype=torch.float32)
y = torch.tensor(y,dtype=torch.long)

lossfn = nn.CrossEntropyLoss() #internal use softmax 
optimizer = optim.Adam(model.parameters(),lr=0.01)

for _ in range(30):
    optimizer.zero_grad()
    yp=model(X)
    loss=lossfn(yp,y)
    loss.backward()
    optimizer.step()
    print(loss)


import cv2
img = cv2.imread(r'C:\Emphasis\FirstProject\CV\img\download.jpg')
print(img.shape)
# Convert to grayscale and resize to 8x8 to match training data
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print(gray.shape)
rs = cv2.resize(gray, (8, 8))
print(rs.shape)
far = rs.flatten()
fimg = torch.tensor(far,dtype=torch.float32)
yp = model(fimg)
print(yp)
fx=torch.softmax(model(fimg),dim=1)
print(torch.argmax(fx,dim=1))
