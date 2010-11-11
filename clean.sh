#!/usr/bin/bash
#
echo Cleaning project

rm -rf sdist
rm -f *.bz2
cd src
./clean.sh
cd core
./clean.sh
