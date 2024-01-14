from ...backend.numpy_backend import NumpyBackend


class NumpyBackend1D(NumpyBackend):
    @classmethod
    def subsample_fourier(cls, x, k):

        cls.complex_check(x)

        y = x.reshape(-1, k, x.shape[-1] // k)

        res = y.mean(axis=-2)

        return res

    @classmethod
    def pad(cls, x, pad_left, pad_right):

        if (pad_left >= x.shape[-1]) or (pad_right >= x.shape[-1]):
            raise ValueError('Indefinite padding size (larger than tensor).')

        paddings = ((0, 0),) * len(x.shape[:-1])
        paddings += (pad_left, pad_right),

        output = cls._np.pad(x, paddings, mode='reflect')

        return output

    @staticmethod
    def unpad(x, i0, i1):

        return x[..., i0:i1]

    @classmethod
    def rfft(cls, x):
        cls.real_check(x)

        return cls._np.fft.fft(x)

    @classmethod
    def irfft(cls, x):
        cls.complex_check(x)

        return cls._fft.ifft(x).real

    @classmethod
    def ifft(cls, x):
        cls.complex_check(x)

        return cls._fft.ifft(x)

backend = NumpyBackend1D
