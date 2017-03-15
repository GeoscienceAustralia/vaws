#!/bin/sh

echo "Running some sims"

for i in {1..1}
do
	python simulation.py -s ../../scenarios/test_scenario1.cfg -o ../../outputs/output
	echo $i
done

