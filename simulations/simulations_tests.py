import unittest
import simulations.tools as st
import numpy as np


class TestShiftDomain(unittest.TestCase):
    def setUp(self):
        pass

    def test_no_change(self):
        x_min = 0
        x_max = 1
        array = np.zeros(3)
        result = st.shift_domain(array, x_min, x_max, x_min, x_max)
        self.assertTrue((result[1] == array).all())

    def test_shrink_left(self):
        x_min = 0
        x_max = 1
        array = np.arange(3)
        result = st.shift_domain(array, x_min, x_max, x_min, 0.5)
        self.assertTrue((result[1] == array[:-1]).all())

    def test_shrink_right(self):
        x_min = 0
        x_max = 1
        array = np.arange(3)
        result = st.shift_domain(array, x_min, x_max, 0.5, x_max)
        self.assertTrue((result[1] == array[1:]).all())

    def test_length(self):
        result = st.shift_domain(np.zeros(10), -1, 2, 3, 6.5)
        self.assertEqual(result[0].shape, result[1].shape)


class TestShiftDomainCL(unittest.TestCase):

    def test_no_change(self):
        x_min = 0
        x_max = 1
        array = np.zeros(3)
        result = st.shift_domain_const_length(array, x_min, dx=(x_max-x_min)/(len(array-1)), new_x_min=x_min, new_length=len(array))
        self.assertTrue((result[1] == array).all())

    def test_length(self):
        length=6
        result = st.shift_domain_const_length(np.zeros(10), -1, 0.1, 3, length)
        self.assertEqual(result[0].shape, (length,))
        self.assertEqual(result[1].shape, (length,))
        self.assertEqual(result[0].shape, result[1].shape)

    def test_dx(self):
        dx = 0.1
        result = st.shift_domain_const_length(np.zeros(10), -1, dx=dx, new_x_min=3, new_length=6)
        self.assertTrue(np.isclose(result[0][1]-result[0][0], dx))


class TestFrontInterferenceTime(unittest.TestCase):

    def setUP(self):
        pass

    def test_dirac(self):
        t = np.linspace(-1, 1, 5)
        signal = np.zeros_like(t)
        signal[3] = 1
        c = 1
        z, corr, offset = st.front_interference_mirror(signal, z0=0, dt=(t[1] - t[0]), c=c, depth=10)
        self.assertEqual(offset, 1.0)
        self.assertEqual(corr[0], 0.5)
        self.assertEqual(corr[1], 0)
        self.assertEqual(z[1]-z[0], (t[1]-t[0])*c/2)
        self.assertEqual(z[40], 10)

    def test_dirac2(self):
        t = np.linspace(-1, 1, 5)
        signal = np.zeros_like(t)
        signal[2] = 1
        z, corr, offset = st.front_interference_mirror(signal, z0=0, dt=(t[1] - t[0]), c=1, depth=10, pulses=2, period=1)
        self.assertEqual(offset, 4.0)
        self.assertEqual(corr[0], 1)
        self.assertEqual(corr[4], 0.5)
        self.assertEqual(corr[1], 0)
        self.assertEqual(z[40], 10)


class TestFrontInterferenceMirror(unittest.TestCase):
    def setUP(self):
        pass

    def test_match_FIT(self):
        wave = np.arange(5)
        z0 = 0
        z_min = 0
        z_max = 1
        expected = st.front_interference_time(wave, z0, wave, z0, dt=0.25, c=1, z_min=z_min, z_max=z_max)
        result =  st.front_interference_mirror(wave, z0, dt=0.25, c=1, depth=z_max)
        self.assertTrue((result[1] == expected[1]).all())
        self.assertTrue((result[0] == expected[0]).all())
        self.assertEqual(result[2], expected[2])

    def test_match_FIT_2(self):
        wave = np.arange(5)
        z0 = 0
        z_min = 0
        z_max = 1
        pulses = 2
        period = 6
        expected = st.front_interference_time(wave, z0, wave, z0, dt=0.25, c=1, z_min=z_min, z_max=z_max, pulses1=
        pulses, pulses2=pulses, period2=period, period1=period)
        result =  st.front_interference_mirror(wave, z0, dt=0.25, c=1, depth=z_max, pulses=pulses, period=period)
        self.assertTrue((result[1] == expected[1]).all())
        self.assertTrue((result[0] == expected[0]).all())
        self.assertEqual(result[2], expected[2])

    def test_z0_dependence(self):
        wave = np.arange(5)
        z0 = 0.5
        z_min = 0
        z_max = 1
        pulses = 2
        period = 6
        expected = result =  st.front_interference_mirror(wave, z0, dt=0.25, c=1, depth=z_max, pulses=pulses, period=period)
        result =  st.front_interference_mirror(wave, z0=1, dt=0.25, c=1, depth=z_max, pulses=pulses, period=period)
        self.assertTrue((result[1] == expected[1]).all())
        self.assertTrue((result[0] == expected[0]).all())
        self.assertEqual(result[2], expected[2])

if __name__ == '__main__':
    unittest.main()
