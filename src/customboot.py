import sys, traceback
from PyQt4.QtGui import QMessageBox

def dlgexcept(exctype, value, tb):
    msg = 'A fatal error has occured: \n\n'
    for entry in traceback.format_exception(type, value, tb):
        msg += entry
    QMessageBox.critical(None, "VAWS Program Error", unicode(msg))
    print "VAWS Program Error: %s" % msg

def debugexcept(type, value, tb):
    if hasattr(sys, 'ps1') or not (sys.stderr.isatty() and sys.stdin.isatty()) or type == SyntaxError:
        sys.__excepthook__(type, value, tb)
    else:
        import pdb
        traceback.print_exception(type, value, tb)
        print
        pdb.pm()
      
# the user must have permissions to these files to prevent the default py2exe wrapper from exitting the app.
import ctypes.wintypes
sys.excepthook = dlgexcept
buf = ctypes.create_unicode_buffer(ctypes.wintypes.MAX_PATH)
ctypes.windll.shell32.SHGetFolderPathW(0, 5, 0, 0, buf)
mydocspath=str(buf.value)
sys.stdout = open('%s\\vaws_stdout.log' % mydocspath, 'w')
sys.stderr = open('%s\\vaws_stderr.log' % mydocspath, 'w')