from simulations.tools import *
import numpy as np
import matplotlib.pyplot as plt
import submodules.lippmann.multilayer_optics_matrix_theory as mt

plt.rcParams["figure.figsize"] = (5, 3)

experiment_number = 0
directory = "./results/"

N = 1000
n0 = 1.45
dn = 1e-3
c0 = 299792458
c = c0 / n0
N_omegas = 300
delta_z = 1E-9
shift = 2E-10 # in meters
phase = 0  # in radians
max_depth = 5E-8
lambdas = wavelengths_omega_spaced()

print("shift in phase =", shift*2*np.pi*c0) # TODO (michalina) add random phase shift
spectrum = mt.generate_gaussian_spectrum(lambdas=lambdas, mu=550E-9, sigma=5E-9)
# spectrum = mt.generate_mono_spectrum(lambdas)
# spectrum = mt.generate_rect_spectrum(lambdas)
a = np.power(spectrum, 1/2)
packet = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(a)))

plt.figure()
plt.plot(lambdas, np.real(spectrum))
plt.title('Original object spectrum')
plt.savefig(directory + "spectrum" + str(experiment_number) + ".pdf")
plt.show()

all_ns = []
depths = 0

for _ in range(10):
    phase = 2*np.pi*np.random.uniform()
    a2 = a*np.exp(1j*+phase)
    spectrum = np.power(np.abs(a), 2)
    packet2 = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(a2)))

    depths, delta_intensity, _ = front_interference_time(packet, 0, packet2, 0, dt=delta_z/(2*c), c=c, z_min=-max_depth, z_max=max_depth)

    ns = delta_intensity - np.min(delta_intensity)
    ns = ns / (np.max(ns))
    all_ns.append(ns)

all_ns = np.array(all_ns)

plt.figure()
plt.plot(depths, np.real(all_ns[0]), label="single pattern")
plt.plot(depths, np.mean(np.real(all_ns), 0), label="mean pattern")
plt.title('Interference pattern')
# plt.axhline(min_cut, c="C1", ls="--", label="lower cutoff")
# plt.axhline(max_cut, c="C1", ls="-.", label="upper cutoff")
plt.legend()
plt.savefig(directory + "interference" + str(experiment_number) + ".pdf")
plt.show()
