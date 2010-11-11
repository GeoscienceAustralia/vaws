from distutils.core import setup
import py2exe
import sys; sys.argv.append('py2exe')
import matplotlib
import glob

opts = {
    'py2exe': { "includes" : ["sip", "PyQt4", "matplotlib.backends",  "matplotlib.backends.backend_qt4agg",
                               "matplotlib.figure","pylab", "numpy", "matplotlib.numerix.fft",
                               "matplotlib.numerix.linear_algebra", "matplotlib.numerix.random_array",
                               "matplotlib.backends.backend_tkagg",
                               "sqlalchemy", "sqlite3"],
                'excludes': ["_gtkagg", "_tkagg", "_agg2", "_cairo", "_cocoaagg", "_fltkagg", "_gtk", "_gtkcairo", 
                             "pywin", "pywin.debugger", "pywin.debugger.dbgcon", "pywin.dialogs", "pywin.dialogs.list", "Tkconstants","Tkinter","tcl",
                             "_imagingtk", "PIL._imagingtk", "ImageTk", "PIL.ImageTk", "FixTk",
                             "_ssl", "difflib", "pdb", "doctest"],
                "packages": ["sqlalchemy.databases.sqlite"],
                'dll_excludes': ['libgdk-win32-2.0-0.dll',
                                 'libgobject-2.0-0.dll',
                                 'msvcr71.dll']
              }
       }

data_files = [(r'mpl-data', glob.glob(r'C:\Python25\Lib\site-packages\matplotlib\mpl-data\*.*')),
                  (r'mpl-data', [r'C:\Python25\Lib\site-packages\matplotlib\mpl-data\matplotlibrc']),
                  (r'mpl-data\images',glob.glob(r'C:\Python25\Lib\site-packages\matplotlib\mpl-data\images\*.*')),
                  (r'mpl-data\fonts',glob.glob(r'C:\Python25\Lib\site-packages\matplotlib\mpl-data\fonts\*.*'))]

setup(name="WindSim",
      version="1.0.4",
      description="Component Based Wind Simulation Prototype",
      author="GeoScience Australia", 
      console=[{"script" : 'damage.py'}], 
      options=opts,
      data_files=data_files)
        

