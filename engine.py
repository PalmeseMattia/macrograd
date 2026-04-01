from typing import Callable, Union
from numbers import Number
import random

class Value:
    data: float
    grad: float
    _backward: Callable[[], None]
    _prev: set["Value"]
    _op: str

    def __init__(self, data ,_children = (), _op = "") -> None:
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    @staticmethod
    def random() -> "Value" :
        return Value(random.uniform(-1, 1))

    def __add__(self, other : "Value | Number"):
        other = other if isinstance(other, Value) else Value(other) 
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward

        return out

    def __mul__(self, other : "Value | Number"):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")
        
        def _backward():
            self.grad  += out.grad * other.data
            other.grad += out.grad * self.data
        out._backward = _backward
        
        return out
    
    def relu(self):
        out = Value(0 if self.data < 0 else self.data, (self, ), "ReLU")

        def _backward():
            self.grad += out.grad * (out.data > 0)
        out.backward = _backward

        return out
    
    def __repr__(self) :
        return f"<Value | Data: {self.data} | Grad: {self.grad}{(" | Op: `" + self._op + "`") if self._op != "" else ""}>"

    def backward(self):
        topo = []
        visited = set()

        def build_topo(v: Value) :
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        self.grad = 1
        for v in reversed(topo):
            v._backward()

class Common:
    def parameters(self) -> list[Value]:
        return []
    
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0
    

class Neuron(Common):
    weights_in : list[Value]
    bias : Value
    non_linear : bool

    def __init__(self, nin : int, non_linear : bool = True):
        self.weights_in = [Value.random() for _ in range (nin)]
        self.bias = Value(0)
        self.non_linear =  non_linear

    def __call__(self, x):
        out = sum((wi * xi for wi,xi in zip(self.weights_in, x)), self.bias)
        return out.relu() if self.non_linear else out

    def parameters(self):
        
    
    def __repr__(self):
        return f"<Node | Weights: {", ".join(str(n) for n in self.weights_in)} | Bias: {self.bias} | Lin: {self.non_linear}>"
    
class Layer:
    neurons : list[Neuron]

    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range (nout)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def __repr__(self):
        return f"Layer: {", ".join(str(n) for n in self.neurons)}"
        