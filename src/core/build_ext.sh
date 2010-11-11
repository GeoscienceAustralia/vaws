#!/usr/bin/bash
#
echo Building engine PYTHON extension PYD...

rm -f *.pyd
rm -f *.o
rm -rf build/*

python ext-setup.py build -c mingw32

cp build/lib.win32-2.6/*.pyd .



