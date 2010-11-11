from distutils.core import setup, Extension

module1 = Extension('engine', 
		sources = ['engine.c'],
		extra_compile_args = ['-O3'],
		include_dirs=['gsl-1.11\\include', 'C:\\temp\\Python26\\Lib\\site-packages\\numpy\\core\\include\\numpy'],
		library_dirs=['gsl-1.11\\lib'],
		libraries=['gsl', 'm'])

setup (name = 'WindSim Engine',
       version = '1.0.1',
       description = 'selected native implementation for speedup',
       ext_modules = [module1])

