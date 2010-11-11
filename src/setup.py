from distutils.core import setup
import py2exe
import sys
import os
sys.argv.append('py2exe')
import matplotlib

print "Hello, world from root setup.py"
opts = {
    'py2exe': { "includes" : ["core.output", "gui.output", "sip", "PyQt4", "matplotlib.backends",  "matplotlib.backends.backend_qt4agg",
                               "matplotlib.figure", "pylab", "numpy", "core.engine",
                               "sqlalchemy", 
                               "sqlite3",
                               "sqlalchemy.databases",
                               "ctypes", "traceback"],
                'excludes': ["_gtkagg", "_tkagg", "_agg2", "_cairo", "_cocoaagg", "_fltkagg", "_gtk", "_gtkcairo", 
                             "pywin", "pywin.debugger", "pywin.debugger.dbgcon", "pywin.dialogs", "pywin.dialogs.list", "Tkconstants","Tkinter","tcl",
                             "_imagingtk", "PIL._imagingtk", "ImageTk", "PIL.ImageTk", "FixTk",
                             "_ssl", "difflib", "pdb", "doctest"],
                "optimize": 1,
                'dll_excludes': ['libgdk-win32-2.0-0.dll',
                                 'libgobject-2.0-0.dll',
                                 'msvcr71.dll', 
								 'MSVCP90.dll']
              }
       }

data_files = [
			  (r'', [r'C:\temp\projects\windtunnel\thirdparty\vcredist\msvcr90.dll',
                     r'C:\temp\projects\windtunnel\thirdparty\vcredist\msvcm90.dll',
                     r'C:\temp\projects\windtunnel\thirdparty\vcredist\msvcp90.dll',
                     r'C:\temp\projects\windtunnel\thirdparty\vcredist\x86_Microsoft.VC90.CRT_1fc8b3b9a1e18e3b_9.0.21022.8_x-ww_d08d0375.manifest',
                     ])
			  ]

data_files += matplotlib.get_py2exe_datafiles()

setup(name="VAWS",
      version=os.getenv("SIM_VER", "0.1"),
      description="Component Based Wind Simulation Prototype",
      author="GeoScience Australia", 
      windows=[{"script" : 'main.pyw'}], 
      options=opts,
	  zipfile=r'lib\library.zip',
      data_files=data_files)
        
