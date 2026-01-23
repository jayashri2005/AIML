import torch
import torch.nn as nn
import numpy as np
from sklearn.datasets import make_moons
import mglearn as mg
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Generate data
x,y=make_moons(noise=0.15,random_state=21)
print(x.shape)

class LogisticRegressionTorch(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))

def train_logreg(model, X, y, lr=0.1, n_iters=1000):
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    criterion = nn.BCELoss()   # Binary NLL
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    losses = []

    for i in range(n_iters):
        optimizer.zero_grad()

        y_pred = model(X)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if i % 100 == 0:
            print(f"Iteration {i:4d} | Loss: {loss.item():.6f}")

    return losses

    
def torch_predict_proba(model, X):
    with torch.no_grad():
        X = torch.tensor(X, dtype=torch.float32)
        return model(X).numpy().ravel()


def torch_predict(model, X, threshold=0.5):
    probs = torch_predict_proba(model, X)
    return (probs >= threshold).astype(int)


model = LogisticRegressionTorch(n_features=2) #creating model
losses = train_logreg(model, x, y) #training model

y_pred = torch_predict(model, x) #predicting
accuracy = np.mean(y_pred == y) #accuracy

print("\nFinal accuracy:", accuracy) #accuracy
print("Learned weights:", model.linear.weight.data) #weights
print("Learned bias:", model.linear.bias.data) #bias

# Create classification visualization
mg.discrete_scatter(x[:,0], x[:,1], y_pred)
plt.legend()
plt.title('PyTorch Logistic Regression Classification')
plt.savefig('pytorch_classification.png')
print("Classification plot saved as 'pytorch_classification.png'")