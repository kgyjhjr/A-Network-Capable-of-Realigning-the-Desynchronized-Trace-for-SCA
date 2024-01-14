import warnings
import re

warnings.filterwarnings('always', category=DeprecationWarning,
                        module=r'^{0}.*'.format(re.escape(__name__)))
warnings.filterwarnings('always', category=PendingDeprecationWarning,
                        module=r'^{0}.*'.format(re.escape(__name__)))
### End Snippet

__all__ = [
            'Scattering1D',
            ]

from .scattering1d import ScatteringEntry1D as Scattering1D



