from ...frontend.base_frontend import ScatteringBase
import math
import numbers
import numpy as np
from warnings import warn

from ..filter_bank import compute_temporal_support, gauss_1d, scattering_filter_factory
from ..utils import (compute_border_indices, compute_padding,
compute_meta_scattering, precompute_size_scattering)


class ScatteringBase1D(ScatteringBase):
    def __init__(self, J, shape, Q=1, T=None, max_order=2, average=None,
                 oversampling=0, out_type='array', backend=None):
        super(ScatteringBase1D, self).__init__()
        self.J = J
        self.shape = shape
        self.Q = Q
        self.T = T
        self.max_order = max_order
        self.average = average
        self.oversampling = oversampling
        self.out_type = out_type
        self.backend = backend

        if average is not None:
            warn(" For average=True, set T=None for default averaging"
                 " or T>=1 for custom averaging."
                 " For average=False set T=0.",
                 DeprecationWarning)

    def build(self):
        self.r_psi = math.sqrt(0.5)
        self.sigma0 = 0.1
        self.alpha = 5.

        # check the number of filters per octave
        if np.any(np.array(self.Q) < 1):
            raise ValueError('Q should always be >= 1, got {}'.format(self.Q))

        if isinstance(self.Q, int):
            self.Q = (self.Q, 1)
        elif isinstance(self.Q, tuple):
            if len(self.Q) == 1:
                self.Q = self.Q + (1, )
            elif len(self.Q) < 1 or len(self.Q) > 2:
                raise NotImplementedError("Q should be an integer, 1-tuple or "
                                          "2-tuple. Scattering transforms "
                                          "beyond order 2 are not implemented.")
        else:
            raise ValueError("Q must be an integer or a tuple")

        # check input length
        if isinstance(self.shape, numbers.Integral):
            self.shape = (self.shape,)
        elif isinstance(self.shape, tuple):
            if len(self.shape) > 1:
                raise ValueError("If shape is specified as a tuple, it must "
                                 "have exactly one element")
        else:
            raise ValueError("shape must be an integer or a 1-tuple")
        N_input = self.shape[0]

        # check T or set default
        if self.T is None:
            self.T = 2 ** self.J
            self.average = True if self.average is None else self.average
        elif self.T > N_input:
            raise ValueError("The temporal support T of the low-pass filter "
                             "cannot exceed input length (got {} > {})".format(
                                 self.T, N_input))
        elif self.T == 0:
            if not self.average:
                self.T = 2 ** self.J
                self.average = False
            else:
                raise ValueError("average must not be True if T=0 "
                                 "(got {})".format(self.average))
        elif self.T < 1:
            raise ValueError("T must be ==0 or >=1 (got {})".format(
                             self.T))
        else:
            self.average = True if self.average is None else self.average
            if not self.average:
                raise ValueError("average=False is not permitted when T>=1, "
                                 "(got {}). average is deprecated in v0.3 in "
                                 "favour of T and will "
                                 "be removed in v0.4.".format(self.T))


        self.log2_T = math.floor(math.log2(self.T))

        # Compute the minimum support to pad (ideally)
        phi_f = gauss_1d(N_input, self.sigma0/self.T)
        min_to_pad = 3 * compute_temporal_support(
            phi_f.reshape(1, -1), criterion_amplitude=1e-3)

        # to avoid padding more than N - 1 on the left and on the right,
        J_max_support = int(np.floor(np.log2(3 * N_input - 2)))
        J_pad = min(int(np.ceil(np.log2(N_input + 2 * min_to_pad))),
                    J_max_support)
        self._N_padded = 2**J_pad

        # compute the padding quantities:
        self.pad_left, self.pad_right = compute_padding(self._N_padded, N_input)
        # compute start and end indices
        self.ind_start, self.ind_end = compute_border_indices(
            self.log2_T, self.J, self.pad_left, self.pad_left + N_input)

    def create_filters(self):
        # Create the filters
        self.phi_f, self.psi1_f, self.psi2_f = scattering_filter_factory(
            self._N_padded, self.J, self.Q, self.T,
            r_psi=self.r_psi, sigma0=self.sigma0, alpha=self.alpha)
        ScatteringBase._check_filterbanks(self.psi1_f, self.psi2_f)

    def meta(self):
        return compute_meta_scattering(
            self.J, self.Q, self.T, self.max_order, self.r_psi, self.sigma0, self.alpha)

    def output_size(self, detail=False):
        size = precompute_size_scattering(self.J, self.Q, self.T,
            self.max_order, self.r_psi, self.sigma0, self.alpha)
        if not detail:
            size = sum(size)
        return size

    def _check_runtime_args(self):
        if not self.out_type in ('array', 'dict', 'list'):
            raise ValueError("The out_type must be one of 'array', 'dict'"
                             ", or 'list'. Got: {}".format(self.out_type))

        if not self.average and self.out_type == 'array':
            raise ValueError("Cannot convert to out_type='array' with "
                             "average=False. Please set out_type to 'dict' or 'list'.")

        if self.oversampling < 0:
            raise ValueError("oversampling must be nonnegative. Got: {}".format(
                self.oversampling))

        if not isinstance(self.oversampling, numbers.Integral):
            raise ValueError("oversampling must be integer. Got: {}".format(
                self.oversampling))

    def _check_input(self, x):
        # basic checking, should be improved
        if len(x.shape) < 1:
            raise ValueError(
                'Input tensor x should have at least one axis, got {}'.format(
                    len(x.shape)))

    @property
    def J_pad(self):
        return int(np.log2(self._N_padded))

    @property
    def N(self):
        return int(self.shape[0])

    _doc_shape = 'N'

    _doc_instantiation_shape = {True: 'S = Scattering1D(J, N, Q)',
                                False: 'S = Scattering1D(J, Q)'}

    _doc_param_shape = \
    ''''''

    _doc_attrs_shape = \
    ''''''

    _doc_param_average = \
    ''''''

    _doc_attr_average = \
    ''''''

    _doc_param_vectorize = \
    ''''''

    _doc_attr_vectorize = \
    ''''''

    _doc_class = \
    ''''''

    _doc_scattering = \
    ''''''
    @classmethod
    def _document(cls):
        instantiation = cls._doc_instantiation_shape[cls._doc_has_shape]
        param_shape = cls._doc_param_shape if cls._doc_has_shape else ''
        attrs_shape = cls._doc_attrs_shape if cls._doc_has_shape else ''

        param_average = cls._doc_param_average if cls._doc_has_out_type else ''
        attr_average = cls._doc_attr_average if cls._doc_has_out_type else ''
        param_vectorize = cls._doc_param_vectorize if cls._doc_has_out_type else ''
        attr_vectorize = cls._doc_attr_vectorize if cls._doc_has_out_type else ''

        cls.__doc__ = ScatteringBase1D._doc_class.format(
            array=cls._doc_array,
            frontend_paragraph=cls._doc_frontend_paragraph,
            alias_name=cls._doc_alias_name,
            alias_call=cls._doc_alias_call,
            instantiation=instantiation,
            param_shape=param_shape,
            attrs_shape=attrs_shape,
            param_average=param_average,
            attr_average=attr_average,
            param_vectorize=param_vectorize,
            attr_vectorize=attr_vectorize,
            sample=cls._doc_sample.format(shape=cls._doc_shape))

        cls.scattering.__doc__ = ScatteringBase1D._doc_scattering.format(
            array=cls._doc_array,
            n=cls._doc_array_n)


__all__ = ['ScatteringBase1D']
