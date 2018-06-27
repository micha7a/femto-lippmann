from simulations.tools import *
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sps


# noinspection PyTypeChecker
def lippmann_transform(omegas, spectrum, depths) -> Tuple[np.ndarray, np.ndarray]:
    """"Compute the Lippmann transform

        lambdas     - vector of wavelengths
        spectrum    - spectrum of light
        depths      - vector of depths

        Returns intensity       - computed intensity of the interfering waves
                delta_intensity - the intensity without the baseline term"""""
    two_k = 2 * omegas / c0

    one_minus_cosines = 0.5 * (1 - np.cos(two_k[None, :] * depths[:, None]))
    cosines = 0.5 * np.cos(two_k[None, :] * depths[:, None])

    intensity = -np.trapz(one_minus_cosines * spectrum[None, :], two_k, axis=1)
    delta_intensity = np.trapz(cosines * spectrum[None, :], two_k, axis=1)
    return intensity, delta_intensity


def simple_filtering(ft, phase):
    half = int(np.floor(len(ft) / 2))
    ft[0:half] = ft[0:half] * np.exp(phase)
    # ft[half] is intentionally left unchanged
    if len(ft) % 2 == 1:
        ft[half + 1:2 * half + 1] = ft[half + 1:2 * half + 1] * np.exp(-phase)
    else:
        ft[half:2 * half] = ft[half:2 * half] * np.exp(-phase)
    return ft


def general_filtering(ft, b):
    assert len(ft) == len(b)
    # b = (b - b[::-1]) / 2


# noinspection PyTypeChecker
def propagation_phase(intensity, omegas, depths, c) -> np.ndarray:
    two_k = 2 * omegas / c
    exponentials = np.exp(-1j * two_k[:, None] * depths[None, :])
    return 1 / np.sqrt(2 * np.pi) * np.trapz(exponentials * intensity[None, :], depths, axis=1)


def propagation_inverse(spectrum, omegas, depths, c) -> np.ndarray:
    two_k = 2 * omegas / c
    exponentials = np.exp(1j * depths[:, None] * two_k[None, :])
    return 1 / np.sqrt(2 * np.pi) * np.trapz(exponentials * spectrum[None, :], two_k, axis=1)


if __name__ == '__main__':
    plt.rcParams["figure.figsize"] = (5, 3)
    reply = input("Save experiment? (y/n) ")
    saving = False
    experiment_number = -1
    if reply == "y":
        saving = True
        experiment_number = input("Experiment name? ")

    directory = "./results/"

    n0 = 1
    c0 = 2 * np.pi
    c = c0 / n0
    N_omegas = 1000
    omegas = np.linspace(-50, 50, N_omegas)
    mu = 30
    sigma = 1

    spectrum = sps.norm(loc=mu, scale=sigma).pdf(omegas)

    plt.figure()
    plt.plot(omegas, np.real(spectrum))
    plt.title('Original object spectrum')
    if saving:
        plt.savefig(directory + "original_spectrum" + str(experiment_number) + ".pdf")
    plt.show()

    delta_z = 0.1
    max_depth = 100

    steps = int(max_depth / delta_z)
    depths = np.linspace(0, max_depth, steps)

    intensity, delta_intensity = lippmann_transform(omegas / n0, spectrum, depths)
    delta_intensity[int(len(delta_intensity) / 2):] = 0

    delta_ns = delta_intensity - np.min(delta_intensity)
    delta_ns = delta_ns / np.max(delta_ns)
    small_ns = (0.1* delta_ns-0.05) + 1
    ns = intensity / np.mean(intensity)
    ns[int(len(intensity) / 2):] = 0

    plt.figure()
    plt.plot(depths, ns, label="pattern")
    plt.title('Interference pattern')
    if saving:
        plt.savefig(directory + "interference_pattern" + str(experiment_number) + ".pdf")
    plt.show()

    pattern_spectrum = propagation_phase(ns, omegas, depths, c0)
    pattern_power_spectrum = np.abs(pattern_spectrum) ** 2
    small_pattern_spectrum = propagation_phase(small_ns, omegas, depths, c0)

    phases = np.linspace(0.05, 0.25, 5)

    fig1, ax1 = plt.subplots(1)
    ax1.plot(depths, ns, label="original")

    fig2, ax2 = plt.subplots(1)
    ax2.plot(omegas, np.real(pattern_power_spectrum), label="original")

    fig3, ax3 = plt.subplots(1)
    omegas_subset = np.arange(mu - 20 * sigma, mu + 20 * sigma, omegas[1] - omegas[0])
    lambdas = 2 * np.pi * c0 / omegas_subset
    r, _ = mt.propagation_arbitrary_layers_Born_spectrum(small_ns, d=delta_z, lambdas=lambdas, plot=False)
    ax3.plot(omegas_subset, r, label="original")

    fig4, ax4 = plt.subplots(1)
    ax4.plot(depths, small_ns, label="original")

    for phase in phases:
        new_pattern = propagation_inverse(simple_filtering(pattern_spectrum, phase * np.pi), omegas, depths, c0)
        ax1.plot(depths, np.abs(new_pattern), label="modified, phase={phi:.2f}pi".format(phi=phase))
        new_pattern_spectrum = propagation_phase(ns, omegas, depths, c0)
        ax2.plot(omegas, np.abs(new_pattern_spectrum) ** 2, label="modified, phase={phi:.2f}pi".format(phi=phase))
        new_small_pattern = propagation_inverse(simple_filtering(small_pattern_spectrum, phase * np.pi), omegas, depths, c0)

        # ax4.plot(depths, new_small_pattern, label="modified, phase={phi:.2f}pi".format(phi=phase))
        # new_r, _ = mt.propagation_arbitrary_layers_Born_spectrum(new_small_pattern, d=delta_z, lambdas=lambdas, plot=False)
        # ax3.plot(omegas_subset, new_r, label="modified, phase={phi:.2f}pi".format(phi=phase))

    ax1.set_title("Interference patterns")
    ax1.legend()

    ax2.set_title("Replayed power spectrum")
    ax2.set_ylim(ymin=0, ymax=1)
    ax2.legend()

    ax3.set_title('Reflected spectrum')
    ax3.set_ylim(ymin=0, ymax=3)
    ax3.legend()

    ax4.set_title("Attenuated patterns")
    ax4.legend()

    if saving:
        fig1.savefig(directory + "interference_patterns" + str(experiment_number) + ".pdf")
        fig2.savefig(directory + "all_spectra" + str(experiment_number) + ".pdf")
        fig3.savefig(directory + "reflected_MT" + str(experiment_number) + ".pdf")
        fig4.savefig(directory + "attenuated_patterns" + str(experiment_number) + ".pdf")
    # plt.ioff()
    plt.show()


    # r, _ = mt.propagation_arbitrary_layers_Born_spectrum(ns, d=delta_z, lambdas=lambdas, plot=False)
    # r_equiv, _ = mt.propagation_arbitrary_layers_Born_spectrum(new_pattern, d=delta_z, lambdas=lambdas, plot=False)
    #
    #
    # plt.plot(lambdas, r, label="almost linear")
    # plt.plot(lambdas, r_equiv, label="new pattern")
    #
    # plt.legend()
