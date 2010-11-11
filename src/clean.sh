#!/usr/bin/bash
#
echo Cleaning src folder

./core/clean.sh
./gui/clean.sh

rm -rf ./build
rm -f diff.txt
rm -f *.exe
