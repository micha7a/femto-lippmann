from simulations.tools import *

# setting process related constants
n0 = 1.45
delta_n = 1e-3
dot_size = 0.2e-6
expected_wavelength = 510e-9
depth = 100e-6
save = True

# frequency discretization
wavelengths = wavelengths_omega_spaced(n=500)

# space discretization
depths_ = np.linspace(0, depth, int(5e4))
depth_res = depths_[1] - depths_[0]

# choose model (small scale - many possible heights, scale close to 1 - one possible height)
block_scale = 0.1
pattern_scale = 2

# generate pattern:
# spectrum = generate_mono_spectrum(wavelengths, color=expected_wavelength)
spectrum = mt.generate_gaussian_spectrum(lambdas=wavelengths, mu=expected_wavelength, sigma=10e-9)
_, delta_intensity = mt.lippmann_transform(wavelengths / n0, spectrum, depths_)
delta_intensity = sigmoid_inverse(delta_intensity)
blocks_, intensities = block_approximate(depths_, pattern_scale * delta_intensity,
                                         math.floor(dot_size / depth_res), block_scale)

# actually calculate things
R, draw_index = simulate_printing(blocks=blocks_,
                                  block_height=intensities,
                                  block_length=dot_size,
                                  max_index_change=delta_n,
                                  z=depths_,
                                  base_index=n0,
                                  lambdas=wavelengths)

# plot the index of refraction (only first part)
range_to_plot = 1000
plt.figure()
plt.plot(depths_[:range_to_plot], draw_index[:range_to_plot])
plt.xlabel("depth", fontsize=20)
plt.ylabel("refractive index", fontsize=20)
plt.ylim([n0, n0 + delta_n * 1.1])

# plot the reflection
plt.figure()
plt.plot(wavelengths, R, linewidth=2)
plt.xlabel(r"$\lambda$", fontsize=20)
plt.ylabel("reflection", fontsize=20)
plt.show()

# how to save file?
code = "results/gaussian_mass_scale_" + str(block_scale)
if save:
    print("saving files in: ", code)
    np.save(code + "_depths", depths_)
    np.save(code + "_idex1", draw_index)
    np.save(code + "_wavels", wavelengths)
    np.save(code + "_reflection1", R)
