import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

print("Random tensor:", torch.randn(5))

# Test autograd: d(ln(x²))/dx = 2/x
X1 = torch.tensor([5.0], dtype=torch.float32, requires_grad=True)
print("X1:", X1)
Y1 = torch.log(X1**2)
Y1.backward()
print("X1.grad:", X1.grad)

# Data: y = 2x + 3 + noise
np.random.seed(0)
x_np = np.linspace(0, 1, 10)
y_np = 2 * x_np + 3 + np.random.normal(0, 0.1, 10)

x = torch.tensor(x_np, dtype=torch.float32)
y = torch.tensor(y_np, dtype=torch.float32)

plt.figure(figsize=(8, 6))
plt.scatter(x, y)

w = torch.randn(1, requires_grad=True)
b = torch.randn(1, requires_grad=True)

learning_rate = 0.01
epochs = 100

losses = []

for epoch in range(epochs):
    y_pred = w * x + b
    loss = torch.mean((y_pred - y) ** 2)
    losses.append(loss.item())
    
    loss.backward()
    
    with torch.no_grad():
        w -= learning_rate * w.grad
        b -= learning_rate * b.grad
    
    w.grad.zero_()
    b.grad.zero_()
    
    # Plot every 10 epochs
    if epoch % 10 == 0:
        plt.plot(x, y_pred.detach(), color='red', alpha=0.6)
        print(f"Epoch {epoch}: Loss = {loss.item():.4f}, w = {w.item():.4f}, b = {b.item():.4f}")

plt.plot(x, y_pred.detach(), color='red', linewidth=3, label='Final fit')
plt.legend()
plt.title('PyTorch Linear Regression GD')
plt.savefig('pytorch_gd_final.png')

print(f"\nTrained model: y = {w.item():.2f}x + {b.item():.2f}")
print("Losses trend:", losses[-5:])  # Last 5 losses


#w-parameter(line) feature selection stocastic gradient descent; w- created out of slopes;w is a slope;b-parameter interms of line moves vertically 