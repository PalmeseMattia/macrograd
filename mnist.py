from TensorEngine import Tensor
import numpy as np

if __name__ == "__main__":
    input = Tensor(np.random.rand(784, 1))
    weights_1 = Tensor(np.random.rand(256, 784))
    first_layer = Tensor(np.empty((1, 256)))
    
    second_layer = Tensor(np.empty((1, 64)))
    
    output = Tensor(np.empty((1,10)))
    
    o = input @ weights_1
    print(o)
    print(o.data.shape)