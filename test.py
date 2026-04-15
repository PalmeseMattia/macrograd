import unittest
import numpy as np
import torch
from TensorEngine import Tensor

class testOperations(unittest.TestCase):
    def test_matmul(self):
        a = np.array([[1,2],[4,5],]).astype(np.float32)
        b = np.array([[1,2],[-3,8],]).astype(np.float32)
        d = np.array([[7,8],[0,-1],]).astype(np.float32)

        # MacroGrad
        a_mg = Tensor(a)
        b_mg = Tensor(b)
        d_mg = Tensor(d)

        c_mg = (a_mg @ b_mg).relu()
        e_mg = (c_mg @ d_mg).relu()
        e_mg.backward(allow_fill=True)

        # PyTorch
        a_pt = torch.tensor(a, requires_grad=True)
        b_pt = torch.tensor(b, requires_grad=True)
        d_pt = torch.tensor(d, requires_grad=True)
        
        c_pt = a_pt.matmul(b_pt).relu()
        e_pt = c_pt.matmul(d_pt).relu()
        e_pt.backward(torch.ones_like(c_pt))

        # Assertions
        np.testing.assert_allclose(e_mg.data, e_pt.detach().numpy())
        np.testing.assert_allclose(a_mg._grad, a_pt.grad.detach().numpy())
        np.testing.assert_allclose(b_mg._grad, b_pt.grad.detach().numpy())

    def test_sub(self):
        a = np.array([[1,2],[4,5],]).astype(np.float32)
        b = np.array([[1,2],[-3,8],]).astype(np.float32)
        
        # Macrograd
        a_mg = Tensor(a)
        b_mg = Tensor(b)
        
        # PyTorch
        a_pt = torch.tensor(a, requires_grad=True)
        b_pt = torch.tensor(b, requires_grad=True)

        c_mg = a_mg - b_mg
        c_pt = a_pt - b_pt

        c_mg.backward(allow_fill=True)
        c_pt.backward(torch.ones_like(c_pt))

        np.testing.assert_allclose(a_mg._grad, a_pt.grad.detach().numpy())
        np.testing.assert_allclose(b_mg._grad, b_pt.grad.detach().numpy())


if __name__ == "__main__":
    unittest.main()