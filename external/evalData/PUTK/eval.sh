#!/bin/bash

echo "trajectory pairs rmse mean median std min max" > "ate.txt"
echo "trajectory pairs rmse mean median std min max rmse mean median std min max" > "rpe.txt"

for i in 1 2 3 4 5 6 7 8
do
	echo $i
	echo "TrajectoryID $i" > "ate"$i".txt"
	echo "TrajectoryID $i" > "rpe"$i".txt"
	python evaluate_ate.py --verbose --plot "ate"$i".svg" "set/Dataset_"$i"_Kin_2/groundtruth.txt" "set/Dataset_"$i"_Kin_2/out.txt" >> "ate"$i".txt"
	cat "ate"$i".txt" | cut -d" " -f 2 | sed ':a;N;$!ba;s/\n/ /g' >> "ate.txt"
	python evaluate_rpe.py --verbose --plot "rpe"$i".svg" --fixed_delta "set/Dataset_"$i"_Kin_2/groundtruth.txt" "set/Dataset_"$i"_Kin_2/out.txt" >> "rpe"$i".txt"
	cat "rpe"$i".txt" | cut -d" " -f 2 | sed ':a;N;$!ba;s/\n/ /g' >> "rpe.txt"
done
