import unittest
from wave import PlanarWave
import numpy as np


class TestPlanarWave(unittest.TestCase):
    def test_creation(self):
        pw = PlanarWave(k=np.array(10), spectrum=np.array(10))
        self.assertEqual(pw.k, np.array(10))

    def test_multiplication(self):
        pw = PlanarWave(k=np.array(10), spectrum=np.array(10))
        alpha = 0.1j
        pwr = pw * alpha
        pwl = alpha * pw
        self.assertAlmostEqual(pw.k, pwr.k)
        self.assertAlmostEqual(pw.k, pwl.k)
        self.assertAlmostEqual(pwr.spectrum, pw.spectrum * alpha)


if __name__ == '__main__':
    unittest.main()
