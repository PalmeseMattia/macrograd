from TensorEngine import Tensor
from tqdm import tqdm
import numpy as np

X = Tensor(np.array([
    [0., 0.],
    [1., 0.],
    [0., 1.],
    [1., 1.]
]))

Y = Tensor(np.array([
    [0.],
    [0.],
    [0.],
    [1.]
]))

W = Tensor(np.random.rand(2,1))
W1 = Tensor(np.random.rand(1,1))
b = Tensor(np.random.rand(1,1))
lr = 0.01

if __name__ == "__main__":
    for i in tqdm(range(10000)):
        y_out = (X @ W @ W1)
        
        # Squared error
        loss = ((Y - y_out) ** 2).mean()
        loss.backward(allow_fill=True)
        print(loss.data)

        W -= W._grad * lr
        W1 -= W1._grad * lr

        W.grad_zero()
        W1.grad_zero()

    print(y_out)

