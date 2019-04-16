from simulations.tools import *
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (5, 3)

experiment_number = 10
directory = "./results/"

N = 1000
n0 = 1.45
dn = 1e-3
c0 = 299792458
c = c0 / n0
N_omegas = 300
delta_z = 1E-9
shift = 2E-14 # in seconds
phase = 0  # in radians
max_depth = 5E-6
lambdas = wavelengths_omega_spaced()
depths = np.arange(0, max_depth, delta_z)

print("shift in phase =", shift*2*np.pi*c0)
spectrum = mt.generate_gaussian_spectrum(lambdas=lambdas, mu=550E-9, sigma=50E-9)
# spectrum = mt.generate_mono_spectrum(lambdas)
# spectrum = mt.generate_rect_spectrum(lambdas)
# a = np.power(spectrum, 1/2)
# a = a + a*np.exp(1j*(shift*2*np.pi*c0/lambdas+phase))
# spectrum = np.power(np.abs(a),2)


plt.figure()
plt.plot(lambdas, np.real(spectrum))
plt.title('Original object spectrum')
plt.savefig(directory + "spectrum" + str(experiment_number) + ".pdf")
plt.show()

intensity, delta_intensity = mt.lippmann_transform(lambdas / n0, spectrum, depths)


# normalize the the intensity
ns = delta_intensity - np.min(delta_intensity)
ns = ns / (np.max(ns))

# below saturation (but with sigmoid)
ns_sigm = sigmoid(ns)
ns_sigm = ns_sigm * dn + n0

# change the power leading to different cut from the saturation
min_cut = 0.5
max_cut = 0.9
ns_sat = sigmoid((ns - min_cut) / (max_cut - min_cut))
ns_sat = ns_sat * dn + n0

plt.figure()
plt.plot(depths, np.real(ns), label="pattern")
plt.title('Interference pattern')
plt.axhline(min_cut, c="C1", ls="--", label="lower cutoff")
plt.axhline(max_cut, c="C1", ls="-.", label="upper cutoff")
plt.legend()
plt.savefig(directory + "interference" + str(experiment_number) + ".pdf")
plt.show()

plt.figure()
plt.plot(depths, ns_sigm, label="almost linear")
plt.plot(depths, ns_sat, label="with saturation")
plt.legend()
plt.title('Refractive index')
plt.savefig(directory + "pattern" + str(experiment_number) + ".pdf")
plt.show()

r, _ = mt.propagation_arbitrary_layers_Born_spectrum(ns_sigm, d=delta_z, lambdas=lambdas, plot=False)
r_sat, _ = mt.propagation_arbitrary_layers_Born_spectrum(ns_sat, d=delta_z, lambdas=lambdas, plot=False)

plt.figure()
plt.plot(lambdas, r, label="almost linear")
plt.plot(lambdas, r_sat, label="with saturation")
plt.title('Reflected spectrum')
plt.legend()
plt.savefig(directory + "reflected" + str(experiment_number) + ".pdf")
plt.show()
