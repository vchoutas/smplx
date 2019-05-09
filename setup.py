import io
import os

from setuptools import find_packages, setup, Command

import subprocess
import shutil
import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Package meta-data.
NAME = 'smplx'
DESCRIPTION = 'PyTorch module for loading the SMPLX body model'
URL = 'http://smpl-x.is.tuebingen.mpg.de'
EMAIL = 'vassilis.choutas@tuebingen.mpg.de'
AUTHOR = 'Vassilis Choutas'
REQUIRES_PYTHON = '>=2.7.0'
VERSION = '0.1.0'

here = os.path.abspath(os.path.dirname(__file__))

try:
    FileNotFoundError
except NameError:
    FileNotFoundError = IOError
# Import the README and use it as the long-description.
# Note: this will only work if 'README.md' is present in your MANIFEST.in file!
try:
    with io.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
        long_description = '\n' + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

# Load the package's __version__.py module as a dictionary.
about = {}
if not VERSION:
    with open(os.path.join(here, NAME, '__version__.py')) as f:
        exec(f.read(), about)
else:
    about['__version__'] = VERSION

setup(name=NAME,
      version=about['__version__'],
      description=DESCRIPTION,
      long_description=long_description,
      long_description_content_type='text/markdown',
      author=AUTHOR,
      author_email=EMAIL,
      python_requires=REQUIRES_PYTHON,
      url=URL,
      packages=['smplx'],
      cmdclass={'build_ext': BuildExtension})
