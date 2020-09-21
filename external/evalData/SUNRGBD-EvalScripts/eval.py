#%matplotlib inline
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
import skimage.io as io
import argparse
import pylab
pylab.rcParams['figure.figsize'] = (10.0, 8.0)

parser = argparse.ArgumentParser(description='Evaluate coco')
parser.add_argument('--category', dest='category', type=int, help='compute only for specific category')
parser.add_argument('--gt', dest='annFile', default='./gtOut.json', help='ground truth file')
parser.add_argument('--dt', dest='resFile', default='./detOut.json', help='detection file')
args = parser.parse_args()

print(args.category)

annType = ['segm','bbox','keypoints']
annType = annType[1]      #specify type here
prefix = 'person_keypoints' if annType=='keypoints' else 'instances'
print('Running for *%s* results.'%(annType))

#initialize COCO ground truth api
#dataDir='./'
#annFile = '%s/gtOut.json'%(dataDir)
cocoGt=COCO(args.annFile)

#initialize COCO detections api
#resFile='%s/detOut.json'
#resFile = resFile%(dataDir)
cocoDt=cocoGt.loadRes(args.resFile)

# running evaluation\n",
cocoEval = COCOeval(cocoGt,cocoDt,annType)
if args.category is not None:
	cocoEval.params.catIds = [args.category]
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()
