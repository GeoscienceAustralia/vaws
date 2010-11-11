#!/usr/bin/bash
#
export SIM_VER=`version.sh`

cd src/core
./build_ext.sh
cd ../..
cd src/gui
./build.sh
cd ../..
./build_srcdist.sh
./build_bindist.sh
./build_installer.sh

echo build fini

