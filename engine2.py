import numpy as np

class Tensor:

    def __init__(self, data, _backward=()):
        self.data = data
        self._backward = lambda: None
        self._grad = 0

    def __matmul__(self, other):
        out = self.data @ other.data

        def backward(self, other):
            self._grad = out._grad @ np.transpose(other.data)
            other._grad = self._grad @ np.transpose(self.data)
        out._backward = backward

        return out

    def __add__(self, other):
        out = self.data + other.data

        def backward(self, other):
            self._grad = out._grad
            other._grad = out._grad
        out._backward = backward

        return out
    
    def __repr__(self):
        return str(a._data)

if __name__ == "__main__":
    a = Tensor(np.array([
            [1,2],
            [4,5],
        ]))

    b = Tensor(np.array([
            [1,2],
            [4,5],
        ]))
    print(str(a))
    print("Transposed:")
    a= np.transpose(a)
    print(str(a))