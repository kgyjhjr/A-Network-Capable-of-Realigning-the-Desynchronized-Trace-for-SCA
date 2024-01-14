import tensorflow as tf

from ...backend.tensorflow_backend import TensorFlowBackend


class TensorFlowBackend1D(TensorFlowBackend):
    @classmethod
    def subsample_fourier(cls, x, k):

        cls.complex_check(x)

        y = tf.reshape(x, (-1, k, x.shape[-1] // k))

        return tf.reduce_mean(y, axis=-2)

    @staticmethod
    def pad(x, pad_left, pad_right):

        if (pad_left >= x.shape[-1]) or (pad_right >= x.shape[-1]):
            raise ValueError('Indefinite padding size (larger than tensor).')

        paddings = [[0, 0]] * len(x.shape[:-1])
        paddings += [[pad_left, pad_right]]

        return tf.pad(x, paddings, mode="REFLECT")

    @staticmethod
    def unpad(x, i0, i1):

        return x[..., i0:i1]

    @classmethod
    def rfft(cls, x):
        cls.real_check(x)

        return tf.signal.fft(tf.cast(x, tf.complex64), name='rfft1d')

    @classmethod
    def irfft(cls, x):
        cls.complex_check(x)

        return tf.math.real(tf.signal.ifft(x, name='irfft1d'))

    @classmethod
    def ifft(cls, x):
        cls.complex_check(x)

        return tf.signal.ifft(x, name='ifft1d')

backend = TensorFlowBackend1D
