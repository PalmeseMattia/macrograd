import unittest
import numpy as np
import torch
from TensorEngine import Tensor

class testOperations(unittest.TestCase):
    def test_matmul(self):
        a = np.array([[1,2],[4,5],]).astype(np.float32)
        b = np.array([[1,2],[4,5],]).astype(np.float32)
        a_macro = Tensor(a)
        b_macro = Tensor(b)
        c_macro = a_macro @ b_macro
        c_macro.backward(allow_fill=True)

        a_torch = torch.tensor(a, requires_grad=True)
        b_torch = torch.tensor(b, requires_grad=True)
        c_torch = a_torch.matmul(b_torch)
        c_torch.backward(torch.ones_like(c_torch))

        np.testing.assert_allclose(c_macro.data, c_torch.detach().numpy())
        np.testing.assert_allclose(a_macro._grad, a_torch.grad.detach().numpy())
        np.testing.assert_allclose(b_macro._grad, b_torch.grad.detach().numpy())

if __name__ == "__main__":
    unittest.main()