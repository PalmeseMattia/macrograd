import numpy as np

class Tensor:

    def __init__(self, data, _backward=(), _parents=()):
        self.data = data
        self._backward = lambda: None
        self._grad = None
        self._parents = _parents

    def __matmul__(self, other):
        out = Tensor(self.data @ other.data, _parents=(self, other))

        def _backward():
            self._grad = out._grad @ np.transpose(other.data)
            other._grad = np.transpose(self.data) @ out._grad
        out._backward = _backward

        return out

    def __add__(self, other):
        out = Tensor(self.data + other.data, _parents=(self, other))

        def _backward():
            self._grad = out._grad
            other._grad = out._grad
        out._backward = _backward

        return out

    def backward(self, allow_fill=False):
        if allow_fill is True:
            self._grad = np.ones_like(self.data)
        #TODO: Build DAG
        #TODO: Call backward recursively
        self._backward()
    
    def __repr__(self):
        return f"Tensor:\n{str(self.data)}\nGrad:\n{str(self._grad)}"
    
if __name__ == "__main__":
    a = Tensor(np.array([1,2,3]))
    b = Tensor(np.array([1,2,3]))
    c = a @ b
    print(a.__hash__)
    print(b.__hash__)
    for n in c._parents:
        print(f'Parent of C: {n.__hash__}')