import numpy as np
import math
import warnings
from scipy.fft import ifft

def adaptive_choice_P(sigma, eps=1e-7):

    val = math.sqrt(-2 * (sigma**2) * math.log(eps))
    P = int(math.ceil(val + 1))
    return P


def periodize_filter_fourier(h_f, nperiods=1):
    N = h_f.shape[0] // nperiods
    v_f = h_f.reshape(nperiods, N).mean(axis=0)
    return v_f


def morlet_1d(N, xi, sigma):
    # Find the adequate value of P<=5
    P = min(adaptive_choice_P(sigma), 5)
    # Define the frequencies over [1-P, P[
    freqs = np.arange((1 - P) * N, P * N, dtype=float) / float(N)
    if P == 1:
        # in this case, make sure that there is continuity around 0
        # by using the interval [-0.5, 0.5]
        freqs_low = np.fft.fftfreq(N)
    elif P > 1:
        freqs_low = freqs
    low_pass_f = np.exp(-(freqs_low**2) / (2 * sigma**2))
    low_pass_f = periodize_filter_fourier(low_pass_f, nperiods=2 * P - 1)
    if xi:
        # define the gabor at freq xi and the low-pass, both of width sigma
        gabor_f = np.exp(-(freqs - xi)**2 / (2 * sigma**2))
        # discretize in signal <=> periodize in Fourier
        gabor_f = periodize_filter_fourier(gabor_f, nperiods=2 * P - 1)
        # find the summation factor to ensure that morlet_f[0] = 0.
        kappa = gabor_f[0] / low_pass_f[0]
        filter_f = gabor_f - kappa * low_pass_f # psi (band-pass) case
    else:
        filter_f = low_pass_f # phi (low-pass) case
    filter_f /= np.abs(ifft(filter_f)).sum()
    return filter_f


def gauss_1d(N, sigma):
    return morlet_1d(N, xi=None, sigma=sigma)


def compute_sigma_psi(xi, Q, r=math.sqrt(0.5)):
    factor = 1. / math.pow(2, 1. / Q)
    term1 = (1 - factor) / (1 + factor)
    term2 = 1. / math.sqrt(2 * math.log(1. / r))
    return xi * term1 * term2


def compute_temporal_support(h_f, criterion_amplitude=1e-3):
    h = ifft(h_f, axis=1)
    half_support = h.shape[1] // 2
    # compute ||h - h_[-N, N]||_1
    l1_residual = np.fliplr(
        np.cumsum(np.fliplr(np.abs(h)[:, :half_support]), axis=1))
    # find the first point above criterion_amplitude
    if np.any(np.max(l1_residual, axis=0) <= criterion_amplitude):
        # if it is possible
        N = np.min(
            np.where(np.max(l1_residual, axis=0) <= criterion_amplitude)[0])\
            + 1
    else:
        # if there is none:
        N = half_support
        # Raise a warning to say that there will be border effects
        warnings.warn('Signal support is too small to avoid border effects')
    return N


def get_max_dyadic_subsampling(xi, sigma, alpha):
    upper_bound = min(xi + alpha * sigma, 0.5)
    j = math.floor(-math.log2(upper_bound)) - 1
    j = int(j)
    return j


def compute_xi_max(Q):
    xi_max = max(1. / (1. + math.pow(2., 3. / Q)), 0.35)
    return xi_max


def compute_params_filterbank(sigma_min, Q, alpha, r_psi=math.sqrt(0.5)):
    xi_max = compute_xi_max(Q)
    sigma_max = compute_sigma_psi(xi_max, Q, r=r_psi)

    if sigma_max <= sigma_min:
        xis = []
        sigmas = []
        elbow_xi = sigma_max
    else:
        xis =  [xi_max]
        sigmas = [sigma_max]

        # High-frequency (constant-Q) region: geometric progression of xi
        while sigmas[-1] > (sigma_min * math.pow(2, 1/Q)):
            xis.append(xis[-1] / math.pow(2, 1/Q))
            sigmas.append(sigmas[-1] / math.pow(2, 1/Q))
        elbow_xi = xis[-1]

    # Low-frequency (constant-bandwidth) region: arithmetic progression of xi
    for q in range(1, Q):
        xis.append(elbow_xi - q/Q * elbow_xi)
        sigmas.append(sigma_min)

    js = [
        get_max_dyadic_subsampling(xi, sigma, alpha) for xi, sigma in zip(xis, sigmas)
    ]
    return xis, sigmas, js


def scattering_filter_factory(N, J, Q, T, r_psi=math.sqrt(0.5),
                              max_subsampling=None, sigma0=0.1, alpha=5., **kwargs):
    # compute the spectral parameters of the filters
    sigma_min = sigma0 / math.pow(2, J)
    Q1, Q2 = Q
    xi1s, sigma1s, j1s = compute_params_filterbank(sigma_min, Q1, alpha, r_psi)
    xi2s, sigma2s, j2s = compute_params_filterbank(sigma_min, Q2, alpha, r_psi)

    # width of the low-pass filter
    sigma_low = sigma0 / T

    # instantiate the dictionaries which will contain the filters
    phi_f = {}
    psi1_f = []
    psi2_f = []

    # compute the band-pass filters of the second order,
    # which can take as input a subsampled
    for (xi2, sigma2, j2) in zip(xi2s, sigma2s, j2s):
        # compute the current value for the max_subsampling,
        # which depends on the input it can accept.
        if max_subsampling is None:
            possible_subsamplings_after_order1 = [j1 for j1 in j1s if j2 > j1]
            if len(possible_subsamplings_after_order1) > 0:
                max_sub_psi2 = max(possible_subsamplings_after_order1)
            else:
                max_sub_psi2 = 0
        else:
            max_sub_psi2 = max_subsampling
        # We first compute the filter without subsampling

        psi_levels = [morlet_1d(N, xi2, sigma2)]
        # compute the filter after subsampling at all other subsamplings
        # which might be received by the network, based on this first filter
        for level in range(1, max_sub_psi2 + 1):
            nperiods = 2**level
            psi_levels.append(periodize_filter_fourier(psi_levels[0], nperiods))
        psi2_f.append({'levels': psi_levels, 'xi': xi2, 'sigma': sigma2, 'j': j2})

    # for the 1st order filters, the input is not subsampled so we
    # can only compute them with N=2**J_support
    for (xi1, sigma1, j1) in zip(xi1s, sigma1s, j1s):
        psi_levels = [morlet_1d(N, xi1, sigma1)]
        psi1_f.append({'levels': psi_levels, 'xi': xi1, 'sigma': sigma1, 'j': j1})

    # compute the low-pass filters phi
    # Determine the maximal subsampling for phi, which depends on the
    # input it can accept (both 1st and 2nd order)
    log2_T = math.floor(math.log2(T))
    if max_subsampling is None:
        max_subsampling_after_psi1 = max(j1s)
        max_subsampling_after_psi2 = max(j2s)
        max_sub_phi = min(max(max_subsampling_after_psi1,
                              max_subsampling_after_psi2), log2_T)
    else:
        max_sub_phi = max_subsampling

    # compute the filters at all possible subsamplings
    phi_levels = [gauss_1d(N, sigma_low)]
    for level in range(1, max_sub_phi + 1):
        nperiods = 2**level
        phi_levels.append(periodize_filter_fourier(phi_levels[0], nperiods))
    phi_f = {'levels': phi_levels, 'xi': 0, 'sigma': sigma_low, 'j': log2_T}

    # return results
    return phi_f, psi1_f, psi2_f
