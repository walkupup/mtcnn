import sys
import os
import random

ratio = 0.8

prefix = '48/neg1/'
suffix = ' 0 -1 -1 -1 -1'
out1 = 'train-neg.txt'
out2 = 'val-neg.txt'
f1 = open(out1, 'w')
f2 = open(out2, 'w')

list = []
for name in os.listdir(r'G:\proj\caffe\mtcnn\mtcnn-walkupup\train\train-pd\48\neg1'):
    name1, ext = os.path.splitext(name)
    if ext == '.jpg' or ext == '.png':
        list.append(prefix + name + suffix)

random.shuffle(list)

for i in range(int(len(list))):
    if i < len(list) * ratio:
        f1.write(list[i] + '\n')
    else:
        f2.write(list[i] + '\n')

f1.close()
f2.close()
