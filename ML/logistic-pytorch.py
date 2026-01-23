import torch 
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mglearn as mg #ml lib
from sklearn.datasets import make_moons #to generate synthetic data

# Generate data
x,y=make_moons(noise=0.15,random_state=21) #data got generated , noise=0.15 make more relastic noise and random_size = ensure reproducibility here x contin 2d coordinates,y cantain labels 0 or 1
print(x.shape)

def sigmoid(z):
    return torch.sigmoid(z)

def negative_log_likelihood(y_true,y_pred,eps=1e-15):
    y_pred = torch.clamp(y_pred,eps,1-eps)
    loss =-torch.mean(y_true*torch.log(y_pred)+(1-y_true)*torch.log(1-y_pred))
    return loss

class LogisticRegressionTorchGD:
    def __init__(self, lr=0.1, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.w = None
        self.b = None
        self.losses = []

    def fit(self, X, y):
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)

        n_samples, n_features = X.shape

        # Initialize parameters
        self.w = torch.zeros(n_features, requires_grad=False)
        self.b = torch.zeros(1, requires_grad=False)

        for i in range(self.n_iters):
            # Forward
            z = X @ self.w + self.b
            y_pred = sigmoid(z)

            # Loss
            loss = negative_log_likelihood(y, y_pred)
            self.losses.append(loss.item())

            # Gradients (manual, same math as NumPy)
            error = y_pred - y
            dw = (1 / n_samples) * (X.T @ error)
            db = (1 / n_samples) * torch.sum(error)

            # Update
            self.w -= self.lr * dw
            self.b -= self.lr * db

            if i % 100 == 0:
                print(f"Iteration {i:4d} | Loss: {loss.item():.6f}")

    def predict_proba(self, X):
        X = torch.tensor(X, dtype=torch.float32)
        return sigmoid(X @ self.w + self.b).detach().numpy()

    def predict(self, X, threshold=0.5):
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)

    def accuracy(self, y_true, y_pred):
        return np.mean(y_true == y_pred)


model = LogisticRegressionTorchGD() #creating model
model.fit(x,y) #training model

y_pred = model.predict(x) #predicting
accuracy = np.mean(y_pred == y) #accuracy

print("\nFinal accuracy:", accuracy) #accuracy
print("Learned weights:", model.w) #weights
print("Learned bias:", model.b) #bias