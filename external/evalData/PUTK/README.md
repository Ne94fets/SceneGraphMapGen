# SLAM Evaluation
Put PUTK data set here. The directory structure should look like:
```
./set/Dataset_*_Kin_2
```

Run in project directory:
```
miracenter domains/Evaluation/etc/PUTK2Eval.xml
```
- Select the PUTK2Eval unit in the top right corner.
- Enter the path to the data set directory as property "Path to data set" in the bottom right corner.
- Set the start property to true and wait til it finished (value will switch back to false)
- Run the following code in `external/evalData/PUTK` to generate detection metrics, output plots and files will be written into same directory:
```
./eval.sh
```
