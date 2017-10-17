#!/usr/bin/bash
#

echo Installing matplotlibsurface Qt Designer plugin into sitepackages...

cp -f matplotsurfacewidget.py /cygdrive/c/Python25/Lib/site-packages/
cp -f matplotsurfaceplugin.py /cygdrive/c/Python25/Lib/site-packages/PyQt4/plugins/designer/python

