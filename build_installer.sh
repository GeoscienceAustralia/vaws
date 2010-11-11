#!/usr/bin/bash
#
export SIM_VER=`version.sh`
echo Building NSIS installer
rm -f *.exe
makensis setup.nsi
mv setup.exe vaws_$SIM_VER.exe

chmod a+x vaws_$SIM_VER.exe


