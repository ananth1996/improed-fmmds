from setuptools import Extension, setup
from Cython.Build import cythonize
import os
from sys import platform
if platform == "darwin":
    print("In MacOS ensure you have gcc-13 installed")
    os.environ["CC"]="gcc-13"
    os.environ["CXX"]="g++-13"

# import numpy as np
# from os.path import dirname, join, abspath
# from numpy.distutils.misc_util import get_info


# inc_path = np.get_include()
# lib_path = [abspath(join(np.get_include(), '..', '..', 'random', 'lib'))]
# lib_path += get_info('npymath')['library_dirs']
# defs = [('NPY_NO_DEPRECATED_API', 0)]

utils_c = Extension("utils_c", 
                    sources = ["utils_c.pyx"],
                    extra_compile_args=['-fopenmp',"-std=c++11"],
                    extra_link_args=['-fopenmp',"-std=c++11"],
                    # include_dirs=[inc_path],
                    )

extensions = [utils_c]



setup(
    name='Utils for Gonzales algorithm',
    ext_modules=cythonize(extensions,annotate=True),
    zip_safe=False,
)