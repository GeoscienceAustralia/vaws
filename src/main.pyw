#!/usr/bin/env python
"""
Program Launcher for GUI Application
"""
import gui.main
import sys

"""
# run our bootscript if we are running as a normal python script (not py2exe)
if not hasattr(sys, "frozen"):
    import customboot
"""

# -----------------------------------------------------------------------------
if __name__ == '__main__':
    gui.main.run_gui('model.db')
