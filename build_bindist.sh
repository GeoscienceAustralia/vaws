#!/usr/bin/bash
#
export RUN_DATE=`date`
export SIM_VER=`version.sh`
echo $RUN_DATE
echo $SIM_VER

cd sdist

echo Cleaning Previous
rm -rf ./dist

echo Making Binary Distribution using PY2EXE...
python setup.py py2exe --custom-boot-script=customboot.py

echo Copying Program Resources...
cd dist
cp *.dll ./lib
cp *.manifest ./lib
mkdir -p gui/images/splash
cp ../gui/images/splash/*.png gui/images/splash
cp ../../release.txt .
cp ../../design/UserGuide.pdf .
cp ../../src/gui/images/home2.ico .
cp ../*.db .
cd ../../..

echo Finished

