from simulations.tools import *
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

n0 = 1.45
dn = 1e-3
c0 = 299792458
c = c0 / n0
delta_z = 1E-9
max_depth = 5E-6
z_shift = max_depth/4 # in meters
shift = z_shift*2/c # in seconds
print("shift in seconds", shift)

lambda_limit=700e-9
n_omegas=500

omega_low = 2 * np.pi * c0 / -lambda_limit
omega_high = 2 * np.pi * c0 / lambda_limit
omegas = np.linspace(omega_low, omega_high, n_omegas)
# lambdas = 2 * np.pi * c0 / omegas

d_omega = omegas[1]-omegas[0]
tlimit = 1.0/d_omega
times = np.linspace(-tlimit/2, tlimit/2, n_omegas)

packet = stats.norm(loc=0, scale=1e-15).pdf(times)*np.exp(1j*1e16*times)
spectrum = np.fft.ifftshift(np.fft.ifft(packet))

fig, ax1 = plt.subplots()
ax1.plot(omegas, np.abs(spectrum))
ax1.set_ylabel("absolute value")
ax2 = ax1.twinx()
ax2.plot(omegas, np.angle(spectrum), "g")
ax2.set_ylabel("angle")
ax2.set_ylim([0,2*np.pi])
plt.title('Original object spectrum')
plt.show()

fig2, ax1 = plt.subplots()
ax1.plot(omegas, np.abs(packet))
ax1.set_ylabel("absolute value")
ax2 = ax1.twinx()
ax2.plot(omegas, np.angle(packet), "g")
ax2.set_ylabel("angle")
ax2.set_ylim([0,2*np.pi])
plt.title('Packet')
plt.show()

depths, delta_intensity, base = front_interference_mirror(packet, z0=0, dt=(times[1]-times[0]), c=c, depth=max_depth, pulses=2, period=shift)

# normalize the the intensity
ns = delta_intensity - np.min(delta_intensity)
ns = ns / (np.max(ns))

# below saturation (but with sigmoid)
ns_sigm = sigmoid(ns)
ns_sigm = ns_sigm * dn + n0

# change the power leading to different cut from the saturation
min_cut = 0.8
max_cut = 0.99
ns_sat = sigmoid((ns - min_cut) / (max_cut - min_cut))
ns_sat = ns_sat * dn + n0

plt.figure()
plt.plot(depths, np.real(ns), label="pattern")
plt.title('Interference pattern')
plt.axhline(min_cut, c="C1", ls="--", label="lower cutoff")
plt.axhline(max_cut, c="C1", ls="-.", label="upper cutoff")
plt.legend()
plt.show()

plt.figure()
plt.plot(depths, ns_sigm, label="almost linear")
plt.plot(depths, ns_sat, label="with saturation")
plt.legend()
plt.title('Refractive index')
plt.show()