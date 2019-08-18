import numpy as np
import unittest

import constants as c
import wave as w


class TestWave(unittest.TestCase):
    def setUp(self) -> None:
        w.PlanarWave.k = c.DEFAULT_K

    def test_creation(self):
        s = w.PlanarWave(spectrum_array=np.zeros(c.OMEGA_STEPS))
        self.assertEqual(np.zeros(c.OMEGA_STEPS).tolist(), s.s.tolist())

    def test_multiplication(self):
        s = w.PlanarWave(spectrum_array=np.ones(c.OMEGA_STEPS))
        alpha = 1 + 0.1j
        sr = s * alpha
        sl = alpha * s
        self.assertEqual(sr, sl)
        self.assertTrue(np.allclose(sr.k, s.k))
        self.assertTrue(np.allclose(sr.s, (s.s * alpha)))

    def test_chirped(self):
        gaussian = w.GaussianPlanarWave()
        not_chirped = w.ChirpedSpectrum()
        chirped = w.ChirpedSpectrum(skew=0.5)
        self.assertEqual(gaussian, not_chirped)
        self.assertNotEqual(gaussian, chirped)

    def test_power(self):
        gaussian = w.GaussianPlanarWave()
        gaussian_power2 = w.GaussianPlanarWave(energy=2)
        self.assertAlmostEqual(c.SINGLE_PULSE_ENERGY, gaussian.total_energy())
        self.assertAlmostEqual(2, gaussian_power2.total_energy())


class TestSigmoid(unittest.TestCase):
    def test_percentile(self):
        perc = 0.01
        self.assertTrue(w.sigmoid(0, 0, 1, percentile=perc) < perc)
        self.assertTrue(w.sigmoid(1, 0, 1, percentile=perc) > 1 - perc)


if __name__ == '__main__':
    unittest.main()
