import numpy as np
import unittest

import constants as c
from wave import PlanarWave


class TestMatrixTheory(unittest.TestCase):
    def setUp(self) -> None:
        PlanarWave.k = np.linspace(2 * np.pi / c.RED, 2 * np.pi / c.VIOLET, c.OMEGA_STEPS)

    def test_boundary_trivial(self):
        n1 = 1
        n2 = 1
        forward = PlanarWave.boundary_matrix(n1, n2)
        self.assertTrue(np.allclose(np.eye(2), forward))
        backward = PlanarWave.boundary_matrix(n1, n2, backward=True)
        self.assertTrue(np.allclose(np.eye(2), backward))

    def test_propagate_trivial(self):
        n = 1
        PlanarWave.k = 1
        dz = 0
        forward = PlanarWave.single_propagation_matrix(n, dz)
        self.assertTrue(np.allclose(np.eye(2), forward))
        backward = PlanarWave.single_propagation_matrix(n, dz, backward=True)
        self.assertTrue(np.allclose(np.eye(2), backward))

    def test_swap_matrices_three_dimensions(self):
        matrix = np.ones((2, 2, 10), dtype=complex)
        new_matrix = PlanarWave.swap_waves(PlanarWave.swap_waves(matrix))
        self.assertTrue(np.allclose(matrix, new_matrix))

    def test_dual_boundary_matrices(self):
        n1 = 1.5
        n2 = 1.2
        forward = PlanarWave.boundary_matrix(n1, n2)
        backward = PlanarWave.boundary_matrix(n1, n2, backward=True)
        self.assertTrue(np.allclose(forward, PlanarWave.swap_waves(backward)))

    def test_dual_propagation_matrices(self):
        n = 1.5
        dz = 0.1
        PlanarWave.k = 7
        forward = PlanarWave.single_propagation_matrix(n, dz)
        backward = PlanarWave.single_propagation_matrix(n, dz, backward=True)
        self.assertTrue(np.allclose(forward, PlanarWave.swap_waves(backward)))

    def test_dual_propagation_matrices_three_dimensions(self):
        n = 1.5
        dz = 0.1
        PlanarWave.k = np.array([7, 8])
        forward = PlanarWave.propagation_matrix(n, dz)
        backward = PlanarWave.propagation_matrix(n, dz, backward=True)
        self.assertTrue(np.allclose(forward, PlanarWave.swap_waves(backward)))

    def test_propagation_matrix_match(self):
        n = 1.5
        dz = 0.1
        PlanarWave.k = 7
        single = PlanarWave.single_propagation_matrix(n, dz)
        PlanarWave.k = np.array([7, 8])
        multiple = PlanarWave.propagation_matrix(n, dz)
        self.assertTrue(np.allclose(single, multiple[:, :, 0]))

    def test_imaginary_index(self):
        n = 1 + 0.1j
        dz = 0.1
        PlanarWave.k = 7
        forward = PlanarWave.single_propagation_matrix(n, dz)
        self.assertTrue(np.abs(forward[0, 0]) < 1)
        self.assertTrue(np.abs(forward[1, 1]) > 1)
        backward = PlanarWave.single_propagation_matrix(n, dz, backward=True)
        self.assertTrue(np.abs(backward[0, 0]) < 1)
        self.assertTrue(np.abs(backward[1, 1]) < 1)
        pass


if __name__ == '__main__':
    unittest.main()
