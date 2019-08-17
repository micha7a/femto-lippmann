import numpy as np
import unittest

import constants as c
import material as m
import wave as w


class TestConstantMaterial(unittest.TestCase):
    z = np.linspace(-5 * c.MICRO, 5 * c.MICRO)

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
        dl = m.ConstantMaterial(z=self.z)
        propagation_data = dl._propagate(s)
        self.assertEqual(propagation_data.shape, (len(s.s), len(self.z)))

    def test_composite(self):
        materials = [m.ConstantMaterial(z=self.z), m.SimpleDielectric(z=self.z, n0=1.5 * np.ones_like(self.z))]
        m.CompositeMaterial(materials)


if __name__ == '__main__':
    unittest.main()
