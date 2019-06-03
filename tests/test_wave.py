import unittest
import wave as w
import numpy as np


class TestSpectrum(unittest.TestCase):
    def test_creation(self):
        s = w.Spectrum(spectrum_array=np.zeros(w.OMEGA_STEPS))
        self.assertEqual(np.zeros(w.OMEGA_STEPS).tolist(), s.s.tolist())

    def test_multiplication(self):
        s = w.Spectrum(spectrum_array=np.ones(w.OMEGA_STEPS))
        alpha = 1 + 0.1j
        sr = s * alpha
        sl = alpha * s
        self.assertEqual(sr, sl)
        self.assertTrue(np.allclose(sr.k, s.k))
        self.assertTrue(np.allclose(sr.s, (s.s * alpha)))

    def test_chirped(self):
        gaussian = w.GaussianSpectrum()
        not_chirped = w.ChirpedSpectrum()
        chirped = w.ChirpedSpectrum(skew=0.5)
        self.assertEqual(gaussian, not_chirped)
        self.assertNotEqual(gaussian, chirped)

    def test_power(self):
        gaussian = w.GaussianSpectrum()
        gaussian_power2 = w.GaussianSpectrum(energy=2)
        self.assertAlmostEqual(w.SINGLE_PULSE_ENERGY, gaussian.total_energy())
        self.assertAlmostEqual(2, gaussian_power2.total_energy())


class TestConstantMaterial(unittest.TestCase):
    z = np.linspace(-5 * w.MICRO, 5 * w.MICRO)

    def test_creation(self):
        dl = w.ConstantMaterial(z=self.z)
        self.assertEqual(1, dl.n0)

    def test_interference(self):
        s_f = w.GaussianSpectrum()
        s_b = w.GaussianSpectrum()
        dl = w.ConstantMaterial(z=self.z)
        empty_space = w.EmptySpace(z=self.z)
        analytic = empty_space.energy_distribution(s_f, s_b)
        computed = dl.energy_distribution(s_f, s_b)
        np.testing.assert_almost_equal(analytic, computed)
        self.assertTrue(np.allclose(analytic, computed))

    def test_propagate(self):
        s = w.GaussianSpectrum()
        dl = w.ConstantMaterial(z=self.z)
        propagation_data = dl._propagate(s)
        self.assertEqual(propagation_data.shape, (len(s.s), len(self.z)))


class TestSigmoid(unittest.TestCase):
    def test_percentile(self):
        perc = 0.01
        self.assertTrue(w.sigmoid(0, 0, 1, percentile=perc) < perc)
        self.assertTrue(w.sigmoid(1, 0, 1, percentile=perc) > 1 - perc)


if __name__ == '__main__':
    unittest.main()
