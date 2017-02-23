#!/bin/sh

echo "Running some sims"

for i in {1..1}
do
	python damage.py -s ../scenarios/carl1.csv --verbose
	echo $i
done

