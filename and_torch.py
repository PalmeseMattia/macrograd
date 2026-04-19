import torch
from tqdm import tqdm
import time

# Data
X = torch.tensor([
    [0., 0.],
    [0., 1.],
    [1., 0.],
    [1., 1.]
])

y = torch.tensor([
    [0.],
    [0.],
    [0.],
    [1.]
])

# Parameters
W = torch.randn(2, 1, requires_grad=True)
b = torch.randn(1, requires_grad=True)

lr = 0.1

for i in tqdm(range(10000), f"Training"):
    y_hat = X @ W + b
    loss = ((y - y_hat) ** 2).mean()
    loss.backward()

    with torch.no_grad():
        W -= W.grad * lr
        b -= b.grad * lr
    
        W.grad.zero_()
        b.grad.zero_()

print(y_hat)