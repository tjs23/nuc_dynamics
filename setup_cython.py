#
# This script can be run using the following command:
#  
#    python setup_cython.py build_ext --inplace
#

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from numpy import get_include as numpy_get_include

numpy_includes = numpy_get_include()

ext_modules = [
    Extension(
        "nuc_dynamics.dyn_util",
        ["nuc_dynamics/dyn_util.pyx"],
        libraries=["m"],
        include_dirs=[numpy_includes]
    )
]

setup(
    name='NucDynamics Cython utilities',
    cmdclass={'build_ext':build_ext},
    ext_modules=ext_modules
)
