from setuptools import Extension, setup
from Cython.Build import cythonize
import os
from sys import platform
if platform == "darwin":
    print("In MacOS ensure you have gcc-13 installed")
    os.environ["CC"]="gcc-13"
    os.environ["CXX"]="g++-13"

parallel_utils = Extension("fmmd.parallel_utils", 
                    sources = ["cython/parallel_utils.pyx"],
                    extra_compile_args=['-fopenmp',"-std=c++11","-O3"],
                    extra_link_args=['-fopenmp',"-std=c++11"],
                    # include_dirs=[inc_path],
                    )

extensions = [parallel_utils]



setup(
    name='Parallel utils for Gonzales algorithm',
    ext_modules=cythonize(extensions,annotate=True),
    zip_safe=False,
)