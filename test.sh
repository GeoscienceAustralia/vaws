#!/usr/bin/bash
#

echo Running baseline tests...

export RUN_DATE=`date`
export SIM_VER=`version.sh`
echo $RUN_DATE
echo $SIM_VER

cd src
rm -rf test/${SIM_VER}/*
mkdir test/${SIM_VER}

echo Setting up database...
cd core
python damage.py -i ../../data/ -m ../test/${SIM_VER}/model.db

echo Running regression...
python damage.py -s ../test/baseline_regress.sce -r ../test/${SIM_VER}/regress.csv -m ../test/${SIM_VER}/model.db

echo Running normal to baseline sim output...
python damage.py -s ../test/baseline_full.sce -o ../test/${SIM_VER} -m ../test/${SIM_VER}/model.db

echo Running under profiler...
python damage.py -s ../test/baseline_full.sce -m ../test/${SIM_VER}/model.db -p > ../test/${SIM_VER}/timing.txt


