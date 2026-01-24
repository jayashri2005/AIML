import numpy as np 
import matplotlib.pyplot as plt 

ar = np.array([3,5,6,2,1,5])

print(np.convolve(ar, [1,2,3])) 

"""

Position 0: 0×1 + 0×2 + 3×3 = 9  → 3 [nupy normalize it using zero padding to get 9 from 3 ]
Position 1: 0×1 + 3×2 + 5×3 = 21 → 11  
Position 2: 3×1 + 5×2 + 6×3 = 31 → 25
Position 3: 5×1 + 6×2 + 2×3 = 23 → 29
Position 4: 6×1 + 2×2 + 1×3 = 13 → 23
Position 5: 2×1 + 1×2 + 5×3 = 19 → 13
Position 6: 1×1 + 5×2 + 0×3 = 11 → 13
Position 7: 5×1 + 0×2 + 0×3 = 5  → 15

output length is len(ar) + len(kernel) - 1 = 6 + 3 - 1 = 8

"""

from sklearn.feature_extraction import image
import torch
from torch import nn
import torch.optim as optim
from sklearn.datasets import load_digits

dt = load_digits()
X,y = dt.data,dt.target
X = torch.tensor(X,dtype=torch.float32).view(-1, 1, 8, 8)
y = torch.tensor(y,dtype=torch.long)

model = nn.Sequential(
    nn.Conv2d(1,32,kernel_size=3,stride=1,padding=1), #8x8
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2), #4x4
    nn.Conv2d(32,64,kernel_size=3,stride=1,padding=1), #4x4
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2), #2x2
    nn.Flatten(),
    nn.Linear(64 * 2 * 2, 128), #128
    nn.ReLU(),
    nn.Linear(128, 10) #10
)

loss_fn = nn.CrossEntropyLoss()
opt = optim.Adam(model.parameters(), lr=0.001)
c=0
for epoch in range(100):
    c=c+1
    opt.zero_grad()
    yp=model(X)
    loss=loss_fn(yp,y)
    loss.backward()
    opt.step()
    if c%10==0:
        print(loss.item())
yyp=model(X)
sfx=torch.softmax(yyp,axis=1)
yp=torch.argmax(sfx,axis=1)
print(yp)

