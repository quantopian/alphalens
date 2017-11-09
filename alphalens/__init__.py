from . import performance
from . import plotting
from . import tears
from . import utils

from ._version import get_versions


__version__ = get_versions()['version']
del get_versions

__all__ = ['performance', 'plotting', 'tears', 'utils']
