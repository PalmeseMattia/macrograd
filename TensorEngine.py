import numpy as np
from numbers import Number

class Tensor:

    def __init__(self, data, _backward=(), _parents=()):
        self.data = data
        self._backward = lambda: None
        self._grad = np.zeros_like(data)
        self._parents = _parents

    def __matmul__(self, other):
        out = Tensor(self.data @ other.data, _parents=(self, other))

        def _backward():
            self._grad += out._grad @ np.transpose(other.data)
            other._grad += np.transpose(self.data) @ out._grad
        out._backward = _backward

        return out

    def __add__(self, other):
        out = Tensor(self.data + other.data, _parents=(self, other))

        def _backward():
            self._grad += out._grad
            other._grad += out._grad
        out._backward = _backward

        return out
    
    def __pow__(self, other):
        assert isinstance(other, Number)
        out = Tensor(self.data ** other, _parents=(self, ))

        def _backward():
            self._grad += (other * self.data**(other-1)) * out._grad
        out._backward = _backward

        return out

    def __sub__(self, other):
        out = Tensor(self.data - other.data, _parents=(self, other))

        def _backward():
            self._grad += 1 * out._grad
            other._grad += -1.0 * out._grad
        out._backward = _backward

        return out
    
    def __isub__(self, other):
        if isinstance(other, Tensor):
            self.data -= other.data
        else:
            self.data -=other
        return self

    def sum(self):
        out = Tensor(np.array([self.data.sum()]), _parents=(self,))

        def _backward():
            self._grad += np.ones_like(self.data) * out._grad
        out._backward = _backward

        return out
    
    def __mul__(self, other):
        out = Tensor(self.data * other.data, _parents=(self, other))

        def _backward():
            self._grad += other.data * out._grad
            other._grad += self.data * out._grad
        out._backward = _backward

        return out
    
    def mean(self):
        div = Tensor(np.array([1/self.data.size]))
        return self.sum() * div

    def relu(self):
        out = Tensor(np.maximum(self.data, np.zeros_like(self.data)))

        def _backward():
            self._grad += (out.data > 0) * out._grad
        out._backward = _backward

        return out

    def backward(self, allow_fill=False):
        if allow_fill is True:
            self._grad = np.ones_like(self.data)
        
        #Build DAG
        visited = set()
        topo = []
        def build_topo(node):
            if node not in visited:
                visited.add(node)
                for n in node._parents:
                    build_topo(n)
                topo.append(node)
        build_topo(self)

        for node in reversed(topo):
            node._backward()

    def grad_zero(self):
        self._grad = np.zeros_like(self.data)
    
    def __repr__(self):
        return f"Tensor:\n{str(self.data)}\nGrad:\n{str(self._grad)}"