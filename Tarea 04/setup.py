from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext
import numpy as np

# ext_module = cythonize("TestOMP.pyx")

ext_module = Extension(
        "Programa",
        ["Programa.pyx"],
)

setup(
        cmdclass = { 'build_ext': build_ext },
        ext_modules = [ext_module],
        include_dirs = [np.get_include()]
)