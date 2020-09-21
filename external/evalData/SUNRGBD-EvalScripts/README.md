# Object Detection and Object Position Estimation Evaluation
Put SUNRGBD kinect v2 data set into this directory.

Then run in project directory:
```
miracenter domains/Evaluation/etc/SUNRGBDEval.xml
```
- Select the SUNRGBDEval unit in the top right corner.
- Enter the path to the data set directory as property "Path to data set" in the bottom right corner.
- Set the start property to `true` and wait til it finished (value will switch back to `false`)
- Run the following code in `external/evalData/SUNRGBD-EvalScripts` to generate detection metrics, output will be stored in `evalOut.txt`:
```
./eval.sh <gtFile> <detFile>
```
- Run the following code to evaluate object position estimation, output will be stored in `rmseOut.txt`:
```
python eval3D.py <rmseFile>
```
