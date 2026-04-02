import numpy as np

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

    def relu(self):
        out = Tensor(np.maximum(self.data, np.zeros_like(self.data)))

        def _backward():
            pass

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
        
    
    def __repr__(self):
        return f"Tensor:\n{str(self.data)}\nGrad:\n{str(self._grad)}"