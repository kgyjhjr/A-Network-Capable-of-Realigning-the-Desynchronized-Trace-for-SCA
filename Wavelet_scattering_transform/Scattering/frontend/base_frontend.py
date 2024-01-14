import importlib
import itertools


class ScatteringBase():
    def __init__(self):
        super(ScatteringBase, self).__init__()

    def build(self):
        raise NotImplementedError

    def _check_filterbanks(psi1s, psi2s):
        assert all((psi1["xi"] < 0.5/(2**psi1["j"])) for psi1 in psi1s)
        psi_generator = itertools.product(psi1s, psi2s)
        condition = lambda psi1_or_2: psi1_or_2[0]['j'] < psi1_or_2[1]['j']
        implication = lambda psi1_or_2: psi1_or_2[0]['xi'] > psi1_or_2[1]['xi']
        assert all(map(implication, filter(condition, psi_generator)))

    def _instantiate_backend(self, import_string):

        if isinstance(self.backend, str):
            if self.backend.startswith(self.frontend_name):
                try:
                    self.backend = importlib.import_module(import_string + self.backend + "_backend", 'backend').backend
                except ImportError:
                    raise ImportError('Backend ' + self.backend + ' not found!')
            else:
                raise ImportError('The backend ' + self.backend + ' can not be called from the frontend ' +
                                   self.frontend_name + '.')

        else:
            if not self.backend.name.startswith(self.frontend_name):
                raise ImportError('The backend ' + self.backend.name + ' is not supported by the frontend ' +
                                   self.frontend_name + '.')

    def create_filters(self):
        raise NotImplementedError
