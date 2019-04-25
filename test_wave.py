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


class TestInterference(unittest.TestCase):
    def test_creation(self):
        s = w.GaussianSpectrum()
        itf = w.Interference(s, s)
        self.assertEqual(itf.forward, itf.backward)
        itf.forward *= 2
        self.assertNotEqual(itf.forward, itf.backward)


class TestMaterial(unittest.TestCase):
    z = np.linspace(-5*w.MICRO, 5*w.MICRO)

    def test_creation(self):
        dl = w.FixedDielectric(z=self.z)
        self.assertEqual(1, dl.n0)

    def test_interference(self):
        s = w.GaussianSpectrum()
        itf = w.Interference(s, s)
        dl = w.FixedDielectric(z=self.z)
        analytic = itf.intensity(z=self.z)
        computed = dl.energy_distribution(itf)
        np.testing.assert_almost_equal(analytic, computed)
        self.assertTrue(np.allclose(analytic, computed))


if __name__ == '__main__':
    unittest.main()
