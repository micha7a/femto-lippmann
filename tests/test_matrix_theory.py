import unittest
import matrix_theory as mt
import numpy as np


class TestMatrixTheory(unittest.TestCase):
    def test_boundary_trivial(self):
        n1 = 1
        n2 = 1
        forward = mt.boundary_matrix(n1, n2)
        self.assertTrue(np.allclose(np.eye(2), forward))
        backward = mt.boundary_matrix(n1, n2, backward=True)
        self.assertTrue(np.allclose(np.eye(2), backward))

    def test_propagate_trivial(self):
        n = 1
        k = 1
        dz = 0
        forward = mt.single_propagation_matrix(n, dz, k)
        self.assertTrue(np.allclose(np.eye(2), forward))
        backward = mt.single_propagation_matrix(n, dz, k, backward=True)
        self.assertTrue(np.allclose(np.eye(2), backward))

    def test_swap_matrices_three_dimensions(self):
        matrix = np.ones((2, 2, 10), dtype=complex)
        new_matrix = mt.swap_waves(mt.swap_waves(matrix))
        self.assertTrue(np.allclose(matrix, new_matrix))

    def test_dual_boundary_matrices(self):
        n1 = 1.5
        n2 = 1.2
        forward = mt.boundary_matrix(n1, n2)
        backward = mt.boundary_matrix(n1, n2, backward=True)
        self.assertTrue(np.allclose(forward, mt.swap_waves(backward)))

    def test_dual_propagation_matrices(self):
        n = 1.5
        dz = 0.1
        k = 7
        forward = mt.single_propagation_matrix(n, dz, k)
        backward = mt.single_propagation_matrix(n, dz, k, backward=True)
        self.assertTrue(np.allclose(forward, mt.swap_waves(backward)))

    def test_dual_propagation_matrices_three_dimensions(self):
        n = 1.5
        dz = 0.1
        k = np.array([7, 8])
        forward = mt.propagation_matrix(n, dz, k)
        backward = mt.propagation_matrix(n, dz, k, backward=True)
        self.assertTrue(np.allclose(forward, mt.swap_waves(backward)))

    def test_propagation_matrix_match(self):
        n = 1.5
        dz = 0.1
        k = np.array([7, 8])
        single = mt.single_propagation_matrix(n, dz, k[0])
        multiple = mt.propagation_matrix(n, dz, k)
        self.assertTrue(np.allclose(single, multiple[:, :, 0]))

    def test_imaginary_index(self):
        n = 1 + 0.1j
        dz = 0.1
        k = 7
        forward = mt.single_propagation_matrix(n, dz, k)
        self.assertTrue(np.abs(forward[0, 0]) < 1)
        self.assertTrue(np.abs(forward[1, 1]) > 1)
        backward = mt.single_propagation_matrix(n, dz, k, backward=True)
        self.assertTrue(np.abs(backward[0, 0]) < 1)
        self.assertTrue(np.abs(backward[1, 1]) < 1)
        pass


if __name__ == '__main__':
    unittest.main()
