import numpy as np

class Tensor:

    def __init__(self, data, _backward=()):
        self.data = data
        self._backward = lambda: None
        self._grad = None

    def __matmul__(self, other):
        out = Tensor(self.data @ other.data)

        def _backward():
            self._grad = out._grad @ np.transpose(other.data)
            other._grad = np.transpose(self.data) @ out._grad
        out._backward = _backward

        return out

    def __add__(self, other):
        out = self.data + other.data

        def _backward():
            self._grad = out._grad
            other._grad = out._grad
        out._backward = _backward

        return out

    def backward(self, allow_fill=False):
        if allow_fill is True:
            self._grad = np.ones_like(self.data)
        self._backward()
    
    def __repr__(self):
        return f"Tensor:\n{str(self.data)}\nGrad:\n{str(self._grad)}"