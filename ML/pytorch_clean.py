import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

print("Random tensor:", torch.randn(5))

# Test autograd: d(ln(x²))/dx = 2/x
X1 = torch.tensor([5.0], dtype=torch.float32, requires_grad=True)
print("X1:", X1)
Y1 = torch.log(X1**2)
Y1.backward()
print("X1.grad:", X1.grad)

# Linear regression with gradient descent
np.random.seed(0)
x_np = np.linspace(0, 1, 10)
y_np = 2 * x_np + 3 + np.random.normal(0, 0.1, 10)

x = torch.tensor(x_np, dtype=torch.float32)
y = torch.tensor(y_np, dtype=torch.float32)

w = torch.randn(1, requires_grad=True)
b = torch.randn(1, requires_grad=True)

learning_rate = 0.01
epochs = 50

plt.figure(figsize=(8, 6))
plt.scatter(x.numpy(), y.numpy(), label='Data')

for epoch in range(epochs):
    y_pred = w * x + b
    loss = torch.mean((y_pred - y) ** 2)
    
    loss.backward()
    
    with torch.no_grad():
        w -= learning_rate * w.grad
        b -= learning_rate * b.grad
    
    w.grad.zero_()
    b.grad.zero_()
    
    if epoch % 10 == 0:
        plt.plot(x.numpy(), y_pred.detach().numpy(), color='red', alpha=0.6)
        print(f"Epoch {epoch}: Loss = {loss.item():.4f}, w = {w.item():.4f}, b = {b.item():.4f}")

plt.plot(x.numpy(), (w * x + b).detach().numpy(), color='red', linewidth=3, label='Final fit')
plt.legend()
plt.title('PyTorch Linear Regression')
plt.savefig('pytorch_regression.png')
print(f"\nFinal model: y = {w.item():.2f}x + {b.item():.2f}")
print("Training completed successfully!")
