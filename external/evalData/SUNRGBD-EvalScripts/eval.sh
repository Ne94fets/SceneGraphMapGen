#!/bin/bash

python eval.py --gt $1 --dt $2 > "evalOut.txt"

cats="65 62 81 63 67 72 70"

for c in $cats
do
	python eval.py --gt $1 --dt $2 --category $c >> "evalOut.txt"
done
