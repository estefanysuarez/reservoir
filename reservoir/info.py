__version__ = '0.0.1'

NAME = 'reservoir'
MAINTAINER = 'Laura Suarez'
VERSION = __version__
LICENSE = 'MIT'
DESCRIPTION = 'A pythonic reservoir toolbox'
#DOWNLOAD_URL = 'http://github.com/rmarkello/pyls'

INSTALL_REQUIRES = [
    'numpy>=1.17',
    'scipy>=1.1',
    'scikit-learn==0.19.1',
    'pandas>=0.23.0',
    'matplotlib',
    'netneurotools',    
    'seaborn>=0.8',
    'bctpy==0.5.0',
    'MDP==3.5',
    'tqdm==4.28.1'
]

TESTS_REQUIRE = [
    'pytest',
    'pytest-cov'
]

PACKAGE_DATA = {
}
