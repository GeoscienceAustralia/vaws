from distutils.core import setup, Extension
import numpy
import os
import sys

if sys.platform == "darwin":
    include_gsl_dir = "/usr/local/include/"
    lib_gsl_dir = "/usr/local/lib/"
    numpy_include_path = os.path.join(numpy.get_include(), 'numpy')

elif sys.platform == "win32":
    include_gsl_dir = r"gsl\include"
    lib_gsl_dir = r"gsl\lib"
    numpy_include_path = os.path.join(numpy.get_include())

setup(name='WindSim Engine',
      version='1.0.1',
      description='selected native implementation for speedup',
      ext_modules=[
          Extension("engine",
                    sources=["engine.c"],
                    include_dirs=[include_gsl_dir, numpy_include_path],
                    library_dirs=[lib_gsl_dir],
                    libraries=["gsl", "gslcblas"]),
      ],
)
