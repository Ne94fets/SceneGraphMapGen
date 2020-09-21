import numpy as np
import json
import math
import argparse

parser = argparse.ArgumentParser(description='Evaluate RMSE')
parser.add_argument('rmseFile', default='./rmse.json', help='rmse text file contianing distances')
args = parser.parse_args()

print(args.rmseFile)

rmseJSON = json.load(open(args.rmseFile, 'r'))
assert type(rmseJSON)==dict, 'rmse file format {} not supported'.format(type(rmseJSON))

msePerCat = dict()
rmsePerCat = dict()
globDistSum = 0
globDistCnt = 0
for cat in rmseJSON:
	distCnt = len(rmseJSON[cat])
	if distCnt == 0:
		msePerCat[cat] = -1
		rmsePerCat[cat] = -1
		continue
	
	distSum = sum(rmseJSON[cat])
	globDistSum += distSum
	globDistCnt += distCnt
	
	catMSE = distSum / distCnt
	catRMSE = math.sqrt(catMSE)
	msePerCat[cat] = catMSE
	rmsePerCat[cat] = catRMSE
	print(catMSE, catRMSE)

with open("rmseOut.txt", "w") as rmseOut:
	rmseOut.write('x ')
	
	for cat in msePerCat:
		rmseOut.write('{' + '{}'.format(cat) + '} ')
		
	rmseOut.write('all\nMSE ')
	
	for cat in msePerCat:
		rmseOut.write('{} '.format(msePerCat[cat]))
	
	rmseOut.write('{}\nRMSE '.format(globDistSum / globDistCnt))
	
	for cat in msePerCat:
		rmseOut.write('{} '.format(rmsePerCat[cat]))
	
	rmseOut.write('{}\n'.format(math.sqrt(globDistSum / globDistCnt)))
