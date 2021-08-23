import numpy as np
import unittest

import constants as c
from wave import PlanarWave
import propagation as p


class TestMatrixTheory(unittest.TestCase):
    def setUp(self) -> None:
        PlanarWave.k = c.DEFAULT_K

    def test_boundary_trivial(self):
        n1 = 1
        n2 = 1
        forward = p.boundary_matrix(n1, n2)
        self.assertTrue(np.allclose(np.eye(2), forward))
        backward = p.boundary_matrix(n1, n2, backward=True)
        self.assertTrue(np.allclose(np.eye(2), backward))

    def test_propagate_trivial(self):
        n = 1
        k = 1
        dz = 0
        forward = p.single_propagation_matrix(k, n, dz)
        self.assertTrue(np.allclose(np.eye(2), forward))
        backward = p.single_propagation_matrix(k, n, dz, backward=True)
        self.assertTrue(np.allclose(np.eye(2), backward))

    def test_boundary_trivial(self):
        r = 0
        forward = p.reflection_matrix(r)
        self.assertTrue(np.allclose(np.eye(2), forward))
        backward = p.reflection_matrix(r, backward=True)
        self.assertTrue(np.allclose(np.eye(2), backward))

    def test_swap_matrices_three_dimensions(self):
        matrix = np.ones((2, 2, 10), dtype=complex)
        new_matrix = p.swap_waves(p.swap_waves(matrix))
        self.assertTrue(np.allclose(matrix, new_matrix))

    def test_dual_boundary_matrices(self):
        n1 = 1.5
        n2 = 1.2
        forward = p.boundary_matrix(n1, n2)
        backward = p.boundary_matrix(n1, n2, backward=True)
        self.assertTrue(np.allclose(forward, p.swap_waves(backward)))

    def test_dual_reflection_matrices(self):
        r = 0.3
        forward = p.reflection_matrix(r)
        backward = p.reflection_matrix(r, backward=True)
        self.assertTrue(np.allclose(forward, p.swap_waves(backward)))

    def test_dual_propagation_matrices(self):
        n = 1.5
        dz = 0.1
        k = 7
        forward = p.single_propagation_matrix(k, n, dz)
        backward = p.single_propagation_matrix(k, n, dz, backward=True)
        self.assertTrue(np.allclose(forward, p.swap_waves(backward)))

    def test_dual_propagation_matrices_three_dimensions(self):
        n = 1.5
        dz = 0.1
        PlanarWave.k = np.array([7, 8])
        forward = p.propagation_matrix(n, dz)
        backward = p.propagation_matrix(n, dz, backward=True)
        self.assertTrue(np.allclose(forward, p.swap_waves(backward)))

    def test_propagation_matrix_match(self):
        n = 1.5
        dz = 0.1
        PlanarWave.k = np.array([0.1, 1, 10])
        multiple = p.propagation_matrix(n, dz)
        for i, k in enumerate(PlanarWave.k):
            single = p.single_propagation_matrix(k, n, dz)
            self.assertTrue(np.allclose(single, multiple[:, :, i]))

    def test_imaginary_index(self):
        n = 1 + 0.1j
        dz = 0.1
        k = 7
        forward = p.single_propagation_matrix(k, n, dz)
        self.assertTrue(np.abs(forward[0, 0]) < 1)
        self.assertTrue(np.abs(forward[1, 1]) > 1)
        backward = p.single_propagation_matrix(k, n, dz, backward=True)
        self.assertTrue(np.abs(backward[0, 0]) < 1)
        self.assertTrue(np.abs(backward[1, 1]) < 1)
        pass


if __name__ == '__main__':
    unittest.main()
