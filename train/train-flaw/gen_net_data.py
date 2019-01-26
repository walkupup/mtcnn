import sys
import numpy as np
import cv2
import os
import numpy.random as npr
from utils import IoU

stdsize = 20
sample_per_box = 50 # 每个正样本周围采样个数
im_dir = r"d:\data\slw\1"
anno_file = os.path.join(im_dir, "pos.txt")
pos_save_dir = os.path.join(im_dir, str(stdsize), "positive")
part_save_dir = os.path.join(im_dir, str(stdsize), "part")
neg_save_dir = os.path.join(im_dir, str(stdsize), 'negative')
save_dir = os.path.join(im_dir,  str(stdsize))

# 在多个box中求最大的iou
def maxIoU(crop_box, boxes):
    m = 0
    for box in boxes:
        iou = IoU(crop_box, box[np.newaxis, :])
        if iou > m:
            m = iou
    return m

#crop_box 是否包含box
def cover(crop_box, box):
    if crop_box[0] <= box[0] and crop_box[1] <= box[1] and crop_box[2] >= box[2] and crop_box[3] >= box[3]:
        return True
    else:
        return False

def mkr(dr):
    if not os.path.exists(dr):
        os.mkdir(dr)

mkr(save_dir)
mkr(pos_save_dir)
mkr(part_save_dir)
mkr(neg_save_dir)

f1 = open(os.path.join(save_dir, 'pos_' + str(stdsize) + '.txt'), 'w')
f2 = open(os.path.join(save_dir, 'neg_' + str(stdsize) + '.txt'), 'w')
f3 = open(os.path.join(save_dir, 'part_' + str(stdsize) + '.txt'), 'w')
with open(anno_file, 'r') as f:
    annotations = f.readlines()
num = len(annotations)
print("%d pics in total" % num)
p_total = 0 # positive
n_total = 0 # negative
d_total = 0 # dont care
idx = 0

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
    idx += 1
    if idx % 100 == 0:
        print(idx, "images done")
    height, width, channel = img.shape
    neg_num = 0
    while neg_num < num * sample_per_box:
        size = npr.randint(18, min(width, height) / 2)
        nx = npr.randint(0, width - size)
        ny = npr.randint(0, height - size)
        crop_box = np.array([nx, ny, nx + size, ny + size])

        Iou = maxIoU(crop_box, boxes)

        cropped_im = img[ny: ny + size, nx: nx + size, :]
        resized_im = cv2.resize(cropped_im, (stdsize, stdsize), interpolation=cv2.INTER_LINEAR)

        if Iou < 0.3:
            # Iou with all gts must below 0.3
            save_file = os.path.join(neg_save_dir, "%s_%s.jpg" % (im_path,n_total))
            f2.write(str(stdsize) + "/negative/%s_%s" % (im_path, n_total) + ' 0\n')
            cv2.imwrite(save_file, resized_im)
            n_total += 1
            neg_num += 1
    for i, box in enumerate(boxes):
        # box (x_left, y_top, x_right, y_bottom)
        print(i)
        x1, y1, x2, y2 = box
        w = x2 - x1 + 1
        h = y2 - y1 + 1

        # ignore small faces
        # in case the ground truth boxes of small faces are not accurate
        if max(w, h) < 12 or x1 < 0 or y1 < 0:
            continue

        # generate positive examples and part faces
        p_idx = 0
        d_idx = 0
        for j in range(1000):
            size = npr.randint(int(min(w, h) * 1.0), np.ceil(1.5 * max(w, h)))

            # delta here is the offset of box center
            delta_x = npr.randint(-w * 0.2, w * 0.2)
            delta_y = npr.randint(-h * 0.2, h * 0.2)

            nx1 = max(x1 + w / 2 + delta_x - size / 2, 0)
            ny1 = max(y1 + h / 2 + delta_y - size / 2, 0)
            nx2 = nx1 + size
            ny2 = ny1 + size

            if nx2 > width or ny2 > height:
                continue
            crop_box = np.array([nx1, ny1, nx2, ny2])

            offset_x1 = (x1 - nx1) / float(size)
            offset_y1 = (y1 - ny1) / float(size)
            offset_x2 = (x2 - nx1) / float(size)
            offset_y2 = (y2 - ny1) / float(size)

            cropped_im = img[int(ny1) : int(ny2), int(nx1) : int(nx2), :]
            resized_im = cv2.resize(cropped_im, (stdsize, stdsize), interpolation=cv2.INTER_LINEAR)

            box_ = box.reshape(1, -1)
            if IoU(crop_box, box_) >= 0.5 and cover(crop_box, box):
                save_file = os.path.join(pos_save_dir, "%s_%d_%s.jpg"%(im_path, i, p_idx))
                f1.write(str(stdsize)+"/positive/%s_%d_%s"%(im_path, i, p_idx) + ' 1 %f %f %f %f\n'%(offset_x1, offset_y1, offset_x2, offset_y2))
                cv2.imwrite(save_file, resized_im)
                p_idx += 1
            elif IoU(crop_box, box_) >= 0.4:
                save_file = os.path.join(part_save_dir, "%s_%s.jpg"%(im_path, d_idx))
                f3.write(str(stdsize)+"/part/%s_%d_%s"%(im_path, i, p_idx) + ' -1 %f %f %f %f\n'%(offset_x1, offset_y1, offset_x2, offset_y2))
                if d_idx < sample_per_box:
                    cv2.imwrite(save_file, resized_im)
                    d_idx += 1
            if p_idx > sample_per_box:
                p_total += p_idx
                d_total += d_idx
                break

    print("%s images done, pos: %s part: %s neg: %s"%(idx, p_total, d_total, n_total))

f1.close()
f2.close()
f3.close()
