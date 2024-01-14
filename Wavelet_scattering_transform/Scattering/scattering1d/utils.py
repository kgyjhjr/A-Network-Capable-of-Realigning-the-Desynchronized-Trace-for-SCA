import numpy as np
import math
from .filter_bank import scattering_filter_factory, compute_params_filterbank

def compute_border_indices(log2_T, J, i0, i1):
    ind_start = {0: i0}
    ind_end = {0: i1}
    for j in range(1, max(log2_T, J) + 1):
        ind_start[j] = (ind_start[j - 1] // 2) + (ind_start[j - 1] % 2)
        ind_end[j] = (ind_end[j - 1] // 2) + (ind_end[j - 1] % 2)
    return ind_start, ind_end

def compute_padding(N, N_input):

    if N < N_input:
        raise ValueError('Padding support should be larger than the original' +
                         'signal size!')
    to_add = N - N_input
    pad_left = to_add // 2
    pad_right = to_add - pad_left
    if max(pad_left, pad_right) >= N_input:
        raise ValueError('Too large padding value, will lead to NaN errors')
    return pad_left, pad_right


def precompute_size_scattering(J, Q, T, max_order, r_psi, sigma0, alpha):
    sigma_min = sigma0 / math.pow(2, J)
    Q1, Q2 = Q
    xi1s, sigma1s, j1s = compute_params_filterbank(sigma_min, Q1, alpha, r_psi)
    xi2s, sigma2s, j2s = compute_params_filterbank(sigma_min, Q2, alpha, r_psi)

    sizes = [1, len(xi1s)]
    size_order2 = 0
    for n1 in range(len(xi1s)):
        for n2 in range(len(xi2s)):
            if j2s[n2] > j1s[n1]:
                size_order2 += 1

    if max_order == 2:
        sizes.append(size_order2)
    return sizes


def compute_meta_scattering(J, Q, T, max_order, r_psi, sigma0, alpha):
    sigma_min = sigma0 / math.pow(2, J)
    Q1, Q2 = Q
    xi1s, sigma1s, j1s = compute_params_filterbank(sigma_min, Q1, alpha, r_psi)
    xi2s, sigma2s, j2s = compute_params_filterbank(sigma_min, Q2, alpha, r_psi)

    meta = {}

    meta['order'] = [[], [], []]
    meta['xi'] = [[], [], []]
    meta['sigma'] = [[], [], []]
    meta['j'] = [[], [], []]
    meta['n'] = [[], [], []]
    meta['key'] = [[], [], []]

    meta['order'][0].append(0)
    meta['xi'][0].append(())
    meta['sigma'][0].append(())
    meta['j'][0].append(())
    meta['n'][0].append(())
    meta['key'][0].append(())

    for (n1, (xi1, sigma1, j1)) in enumerate(zip(xi1s, sigma1s, j1s)):
        meta['order'][1].append(1)
        meta['xi'][1].append((xi1,))
        meta['sigma'][1].append((sigma1,))
        meta['j'][1].append((j1,))
        meta['n'][1].append((n1,))
        meta['key'][1].append((n1,))

        if max_order < 2:
            continue

        for (n2, (xi2, sigma2, j2)) in enumerate(zip(xi2s, sigma2s, j2s)):
            if j2 > j1:
                meta['order'][2].append(2)
                meta['xi'][2].append((xi1, xi2))
                meta['sigma'][2].append((sigma1, sigma2))
                meta['j'][2].append((j1, j2))
                meta['n'][2].append((n1, n2))
                meta['key'][2].append((n1, n2))

    for field, value in meta.items():
        meta[field] = value[0] + value[1] + value[2]

    pad_fields = ['xi', 'sigma', 'j', 'n']
    pad_len = max_order

    for field in pad_fields:
        meta[field] = [x + (math.nan,) * (pad_len - len(x)) for x in meta[field]]

    array_fields = ['order', 'xi', 'sigma', 'j', 'n']

    for field in array_fields:
        meta[field] = np.array(meta[field])

    return meta
