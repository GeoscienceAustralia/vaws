#!/usr/bin/bash
#
export RUN_DATE=`date`
export SIM_VER=`./version.sh`
echo $RUN_DATE
echo $SIM_VER

echo Creating clean dist folder...
rm -rf ./sdist
mkdir sdist
mkdir sdist/gui
mkdir sdist/gui/ui
mkdir sdist/gui/images
mkdir sdist/gui/images/splash
mkdir sdist/gui/help
mkdir sdist/core

echo Copying Python source...
cp src/*.pyw sdist
cp src/*.py sdist
cp src/gui/*.py sdist/gui
cp src/gui/*.pyw sdist/gui
cp src/gui/*.qrc sdist/gui
cp src/gui/ui/*.ui sdist/gui/ui
cp src/gui/images/*.png sdist/gui/images
cp src/gui/images/splash/*.png sdist/gui/images/splash
cp src/gui/help/*.html sdist/gui/help
cp src/core/*.py sdist/core
cp src/core/*.pyd sdist/core

echo Making clean database...
cd sdist/core
python damage.py -i ../../data/ -m ../model.db
rm *.pyc

echo Copying over scenarios...
cd ..
mkdir scenarios
#cp ../src/scenarios/*.csv ./scenarios

echo Building compressed SOURCE archive...
export ARC=../windsim_${SIM_VER}.tar.bz2
rm -f ${ARC}
tar -cjf ${ARC} ./

