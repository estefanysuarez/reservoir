__version__ = '0.0.1'

NAME = 'reservoir'
MAINTAINER = 'Laura Suarez'
VERSION = __version__
LICENSE = 'MIT'
DESCRIPTION = 'A pythonic reservoir toolbox'
#DOWNLOAD_URL = 'http://github.com/rmarkello/pyls'

INSTALL_REQUIRES = [
    'numpy',
    'scikit-learn',
    'scipy',
    'tqdm'
]

TESTS_REQUIRE = [
    'pytest',
    'pytest-cov'
]

PACKAGE_DATA = {
}
