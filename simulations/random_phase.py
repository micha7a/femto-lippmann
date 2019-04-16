from simulations.tools import *
import numpy as np
import matplotlib.pyplot as plt
import submodules.lippmann.multilayer_optics_matrix_theory as mt

plt.rcParams["figure.figsize"] = (5, 3)

experiment_number = 700
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
depths = np.arange(-max_depth, max_depth, delta_z)

# print("shift in phase =", shift*2*np.pi*c0)
spectrum = mt.generate_gaussian_spectrum(lambdas=lambdas, mu=550E-9, sigma=50E-9)
# spectrum = mt.generate_mono_spectrum(lambdas)
# spectrum = mt.generate_rect_spectrum(lambdas)
a = np.power(spectrum, 1/2)


def sft(spectrum, lambdas, depths):
    ks = 2 * np.pi / lambdas
    dk = ks[1]-ks[0]
    exponenets = depths[:, None] @ ks[None, :]
    transform = np.exp(-1j * exponenets)*dk
    print(transform.shape)
    return transform @ spectrum


def isft(pattern, lambdas, depths):
    ks = 2 * np.pi / lambdas
    dz = depths[1]-depths[0]
    exponenets = ks[:, None] @ depths[None, :]
    transform = np.exp(1j * exponenets)*dz/(2 * np.pi)
    print(transform.shape)
    return transform @ pattern


packet = sft(spectrum, lambdas, depths)

plt.figure()
plt.plot(lambdas, np.real(spectrum))
plt.title('Original object spectrum')
plt.savefig(directory + "spectrum" + str(experiment_number) + ".pdf")
plt.show()

plt.figure()
plt.plot(depths, np.real(packet))
plt.title('Packet')
plt.savefig(directory + "packet" + str(experiment_number) + ".pdf")
plt.show()

all_ns = []
all_ns_sat = []
sum_ns_sat = None
repetitions = 1

for _ in range(repetitions):
    phase = 2*np.pi*np.random.uniform()
    a2 = a*np.exp(1j*phase)
    phase3 = 2*np.pi*np.random.uniform()
    a3 = a*np.exp(1j*phase3)

    packet2 = sft(a2, lambdas, depths)
    packet3 = sft(a3, lambdas, depths)

    depths_end, delta_intensity, _ = front_interference_time(packet2, 0, packet3, 0, dt=delta_z/c, c=c, z_min=-max_depth, z_max=max_depth)

    ns = delta_intensity - np.min(delta_intensity)
    ns = ns / (np.max(ns))
    all_ns.append(ns)

    min_cut = 0.5
    max_cut = 0.9
    ns_sat = sigmoid((ns - min_cut) / (max_cut - min_cut))
    ns_sat = ns_sat * dn
    all_ns_sat.append(ns_sat)

all_ns = np.array(all_ns)
all_ns_sat = np.array(all_ns_sat)

plt.figure()
plt.plot(depths_end, np.real(all_ns.T), color='C0', alpha=1/repetitions)
plt.plot(depths_end, np.sum(np.real(all_ns), 0), label="sum of {} patterns".format(repetitions), color="C1")
plt.title('Interference pattern')
# plt.axhline(min_cut, c="C1", ls="--", label="lower cutoff")
# plt.axhline(max_cut, c="C1", ls="-.", label="upper cutoff")
plt.legend(loc=1)
plt.savefig(directory + "interference" + str(experiment_number) + ".pdf")
plt.show()

plt.figure()
plt.plot(depths_end, n0 + np.real(all_ns_sat.T), color='C0', alpha=1/repetitions)
plt.plot(depths_end, n0 + np.sum(np.real(all_ns_sat), 0), label="sum of {} patterns".format(repetitions), color="C1")
plt.title('Recorded pattern ' + str() )
# plt.axhline(min_cut, c="C1", ls="--", label="lower cutoff")
# plt.axhline(max_cut, c="C1", ls="-.", label="upper cutoff")
plt.legend()
plt.savefig(directory + "recorded" + str(experiment_number) + ".pdf")
plt.show()

delta_z = depths_end[1] - depths_end[0]
pattern_length = len(all_ns_sat[0])

r, _ = mt.propagation_arbitrary_layers_Born_spectrum((n0 + np.real(all_ns_sat[0])), d=delta_z, lambdas=lambdas, plot=False)
r_av, _ = mt.propagation_arbitrary_layers_Born_spectrum(n0 + np.sum(np.real(all_ns_sat), 0), d=delta_z, lambdas=lambdas, plot=False)

plt.figure()
plt.plot(lambdas, r, label="single pattern")
plt.plot(lambdas, r_av, label="sum of {} patterns".format(repetitions))
plt.title('Reflected spectrum')
plt.legend()
plt.savefig(directory + "reflected" + str(experiment_number) + ".pdf")
plt.show()
