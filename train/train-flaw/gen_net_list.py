import sys
import os
import random

train_ratio = 0.8
save_dir = "./12"
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
f1 = open(os.path.join(save_dir, 'pos_12.txt'), 'r')
f2 = open(os.path.join(save_dir, 'neg_12.txt'), 'r')
f3 = open(os.path.join(save_dir, 'part_12.txt'), 'r')

pos = f1.readlines()
neg = f2.readlines()
part = f3.readlines()
random.shuffle(pos)
random.shuffle(neg)
random.shuffle(part)
f = open(os.path.join(save_dir, 'label-train.txt'), 'w')
fval = open(os.path.join(save_dir, 'label-val.txt'), 'w')

for i in range(int(len(pos))):
    p = pos[i].find(" ") + 1
    pos[i] = pos[i][:p-1] + ".jpg " + pos[i][p:-1] + "\n"
    if i < len(pos) * train_ratio:
        f.write(pos[i])
    else:
        fval.write(pos[i])

for i in range(int(len(neg))):
    p = neg[i].find(" ") + 1
    neg[i] = neg[i][:p-1] + ".jpg " + neg[i][p:-1] + " -1 -1 -1 -1\n"
    if i < len(neg) * train_ratio:
        f.write(neg[i])
    else:
        fval.write(neg[i])

for i in range(int(len(part))):
    p = part[i].find(" ") + 1
    part[i] = part[i][:p-1] + ".jpg " + part[i][p:-1] + "\n"
    if i < len(part) * train_ratio:
        f.write(part[i])
    else:
        fval.write(part[i])

f1.close()
f2.close()
f3.close()
f.close()
fval.close()
