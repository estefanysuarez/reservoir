__version__ = '0.0.1'

NAME = 'reservoir'
MAINTAINER = 'Laura Suarez'
VERSION = __version__
LICENSE = 'MIT'
DESCRIPTION = 'A pythonic reservoir toolbox'
#DOWNLOAD_URL = 'http://github.com/rmarkello/pyls'

INSTALL_REQUIRES = [
    'numpy>=1.17.0',
    'scipy>=1.1.0',
    'scikit-learn>=0.19.1',
    'pandas>=1.0.5',
    'matplotlib>=3.0.1',
    'netneurotools==0.2.1',
    'seaborn==0.9.0',
    'bctpy==0.5.0',
    'networkx>=2.1',
    'MDP>=3.5',
    'tqdm==4.28.1',
]

TESTS_REQUIRE = [
    'pytest',
    'pytest-cov'
]

PACKAGE_DATA = {
}
