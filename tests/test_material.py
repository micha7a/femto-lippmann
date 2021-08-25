import numpy as np
import unittest

import constants as c
import material as m
import wave as w
import propagation as p


class TestConstantMaterial(unittest.TestCase):
    def setUp(self) -> None:
        self.z = np.linspace(-5 * c.MICRO, 5 * c.MICRO)

    def test_creation(self):
        dl = m.ConstantMaterial(z=self.z)
        self.assertEqual(1, dl.n0)

    def test_interference(self):
        s_f = w.GaussianPlanarWave()
        s_b = w.GaussianPlanarWave()
        dl = m.ConstantMaterial(z=self.z)
        empty_space = m.EmptySpace(z=self.z)
        analytic = empty_space.energy_distribution(s_f, s_b)
        computed = dl.energy_distribution(s_f, s_b)
        np.testing.assert_almost_equal(analytic, computed)
        self.assertTrue(np.allclose(analytic, computed))

    def test_propagate(self):
        s = w.GaussianPlanarWave()
        cm = m.ConstantMaterial(z=self.z)
        propagation_data = cm._propagate(s)
        self.assertEqual(propagation_data.shape, (len(s.s), len(self.z)))


class TestDielectric(unittest.TestCase):
    def setUp(self) -> None:
        self.z = np.linspace(-5 * c.MICRO, 5 * c.MICRO)
        self.materials = [
            m.ConstantMaterial(n0=1.5, z=self.z),
            m.SimpleDielectric(z=self.z, n0=1.5 * np.ones_like(self.z)),
            m.PhotoSensitiveMaterial(z=self.z, n0=1.5, min_energy=1 * c.NANO)
        ]

    def test_energy_response(self):
        s = w.GaussianPlanarWave()
        for material in self.materials:
            energy = material.energy_distribution(s, s)
            deposited = material.energy_response(energy)
            np.testing.assert_array_less(deposited, energy)

    def test_propagation(self):
        s = w.GaussianPlanarWave(mean=c.GREEN, std=(5 * c.MICRO))
        for dec in [10, 6]:
            for material in self.materials:
                material.record(s, s)
            np.testing.assert_array_almost_equal(self.materials[0].recent_energy,
                                                 self.materials[1].recent_energy,
                                                 decimal=dec)
            np.testing.assert_array_almost_equal(self.materials[0].recent_energy,
                                                 self.materials[2].recent_energy,
                                                 decimal=dec)

    def test_single_layer(self):
        idx_z = 1
        dielectric = m.SimpleDielectric(z=self.z, n0=1, name="transfer matrices")
        dielectric._ns = dielectric.index_of_refraction()
        matrix = dielectric.single_layer_matrix(idx_z)
        matrix2 = p.propagation_matrix(dielectric._ns[idx_z - 1], dielectric.z[idx_z] - dielectric.z[idx_z - 1])
        boundary = p.boundary_matrix(dielectric._ns[idx_z - 1], dielectric._ns[idx_z])
        matrix2 = np.einsum('ijk,jl->ilk', matrix2, boundary)
        np.testing.assert_array_almost_equal(matrix, matrix2)


class TestComposite(unittest.TestCase):
    def setUp(self) -> None:
        self.z = np.linspace(-5 * c.MICRO, 5 * c.MICRO, 3)
        self.materials = [
            m.ConstantMaterial(n0=1.5, z=self.z),
            m.SimpleDielectric(z=self.z, n0=1.5 * np.ones_like(self.z))
        ]

    def test_composite(self):
        composite = m.CompositeMaterial(self.materials)
        manual = m.SimpleDielectric(z=np.linspace(0 * c.MICRO, 20 * c.MICRO, 5), n0=1.5)
        self.assertTrue(manual.z.shape == composite.z.shape)
        np.testing.assert_almost_equal(manual.z, composite.z)
        np.testing.assert_almost_equal(manual.material_matrix(w.PlanarWave), composite.material_matrix(w.PlanarWave))


if __name__ == '__main__':
    unittest.main()
