#!/usr/bin/bash
#
echo "Building new model.db"

rm -f ../model.db

python damage.py -m ../model.db -i ../../data/

echo "Done"



