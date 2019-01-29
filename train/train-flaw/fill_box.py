# 正样本填充

import sys
import numpy as np
import cv2
import os
import numpy.random as npr
from utils import IoU

im_dir = r"d:\data\slw\1"
im_out_dir = os.path.join(im_dir, 'filled')
anno_file = os.path.join(im_dir, "pos.txt")

def mkr(dr):
    if not os.path.exists(dr):
        os.mkdir(dr)

mkr(im_out_dir)

with open(anno_file, 'r') as f:
    annotations = f.readlines()
num = len(annotations)

for annotation in annotations:
    annotation = annotation.strip().split(' ')
    im_path = annotation[0]
    num = int(annotation[1])
    if len(annotation) != num * 4 + 2:
        print('wrong annotation')
        continue
    bbox = list(map(int, annotation[2:]))
    boxes = np.array(bbox, dtype=np.float32).reshape(-1, 4)
    img = cv2.imread(os.path.join(im_dir, im_path))
    height, width, channel = img.shape
    for box in boxes:
        img[int(box[1]):int(box[3]), int(box[0]):int(box[2]), :] = 255
    cv2.imwrite(os.path.join(im_out_dir, im_path), img)
    #cv2.waitKey()