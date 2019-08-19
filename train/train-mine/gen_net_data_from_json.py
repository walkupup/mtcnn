import sys
import numpy as np
import cv2
import os
import json
import numpy.random as npr
from utils import IoU
import ellipses

# 在标记多个正样本的大图上进行样本生成
stdsize = 20 # 生成的正样本大小
sample_per_box = 50 # 每个正样本周围采样个数
im_dir = r"D:\data\mine\web"
pos_save_dir = os.path.join(im_dir, str(stdsize), "positive")
part_save_dir = os.path.join(im_dir, str(stdsize), "part")
neg_save_dir = os.path.join(im_dir, str(stdsize), 'negative')
save_dir = os.path.join(im_dir,  str(stdsize))

def json_get_point(json_name):
    """
    从json文件提取目标点的信息。
    Return：
        点信息的list
    """
    with open(json_name, 'r') as load_f:
        load_dict = json.load(load_f)
    pts_total = []
    for shape in load_dict['shapes']:
        pts_total.append(shape['points'])
    return pts_total


# 在多个box中求最大的iou
def maxIoU(crop_box, boxes):
    m = 0
    for box in boxes:
        iou = IoU(crop_box, box[np.newaxis, :])
        if iou > m:
            m = iou
    return m

#crop_box 是否包含box, m 为margin
def cover(crop_box, box, m):
    if crop_box[0] <= box[0] - m and crop_box[1] <= box[1] - m and crop_box[2] >= box[2] + m and crop_box[3] >= box[3] + m:
        return True
    else:
        return False

def mkr(dr):
    if not os.path.exists(dr):
        os.mkdir(dr)

def train_val_split(image_root, train_file, val_file):
    """
    划分训练集和验证集.
        image_root: 原始图像根目录
        train_file, val_file: 输出图像列表txt文件，格式为"图像文件名 类别id"
    """
    ftrain = open(train_file, 'w')
    fval = open(val_file, 'w')
    train_names = []
    val_names = []
    names = os.listdir(image_root)
    #random.shuffle(names)
    names = [os.path.join(image_root, name) + ' 1\n' for name in names if os.path.splitext(name)[1] == '.jpg']
    num = int(len(names) * 0.99) # train的比例
    train_names.extend(names[0:num])
    val_names.extend(names[num:])
    ftrain.writelines(train_names)
    fval.writelines(val_names)
    ftrain.close()
    fval.close()

# 把pts中的前n-1个点拟合成椭圆，再加上最后一点输出；如果点数小于5，不用拟合椭圆，直接输出
def fit_ellipse(pts):
    if pts.shape[0] >= 6:  # circle
        n = pts.shape[0]
        data = [pts[0:n - 1, 0], pts[0:n - 1, 1]]
        lsqe = ellipses.LSqEllipse()
        lsqe.fit(data)
        center, width, height, phi = lsqe.parameters()
        ellpts = cv2.ellipse2Poly((int(center[0]), int(center[1])), (int(width), int(height)), int(phi / 3.14159 * 180),
                                  0, 360, 5)
        ellpts = np.append(ellpts, [[pts[n - 1, 0], pts[n - 1, 1]]], axis=0)
        return ellpts
    else:  # rectangle
        return pts

mkr(save_dir)
mkr(pos_save_dir)
mkr(part_save_dir)
mkr(neg_save_dir)

f1 = open(os.path.join(save_dir, 'pos_' + str(stdsize) + '.txt'), 'w')
f2 = open(os.path.join(save_dir, 'neg_' + str(stdsize) + '.txt'), 'w')
f3 = open(os.path.join(save_dir, 'part_' + str(stdsize) + '.txt'), 'w')

train_val_split(im_dir, 'train.txt', 'val.txt')

f = open('train.txt', 'r')
names = f.readlines()
p_total = 0  # positive
n_total = 0  # negative
d_total = 0  # dont care
idx = 0
for name in names:
    image_name, c = name.strip().split()
    im_path = os.path.basename(image_name)[0:-4]
    json_name, ext = os.path.splitext(image_name)
    json_name = json_name + '.json'
    pts_total = json_get_point(json_name)
    num = len(pts_total)
    img = cv2.imread(os.path.join(im_dir, image_name))
    boxes_ = []
    for pts in pts_total:
        #pts_ = np.array(pts, dtype=np.int32)
        pts_ = fit_ellipse(np.array(pts))
        x1, y1 = np.min(pts_, 0)
        x2, y2 = np.max(pts_, 0)
        boxes_.append([x1, y1, x2, y2])
    boxes = np.array(boxes_, np.int32)

    idx += 1
    if idx % 100 == 0:
        print(idx, "images done")
    height, width, channel = img.shape
    neg_num = 0
    while neg_num < num * sample_per_box * 0.6: # 打了0.6折扣
        size = npr.randint(stdsize, min(width, height) / 2)
        nx = npr.randint(0, width - size)
        ny = npr.randint(0, height - size)
        crop_box = np.array([nx, ny, nx + size, ny + size])

        Iou = maxIoU(crop_box, boxes)
        if Iou < 0.3:
            cropped_im = img[ny: ny + size, nx: nx + size, :]
            resized_im = cv2.resize(cropped_im, (stdsize, stdsize), interpolation=cv2.INTER_LINEAR)
            # Iou with all gts must below 0.3
            save_file = os.path.join(neg_save_dir, "%s_%s.jpg" % (im_path,n_total))
            f2.write(str(stdsize) + "/negative/%s_%s" % (im_path, n_total) + ' 0\n')
            cv2.imwrite(save_file, resized_im)
            n_total += 1
            neg_num += 1
    for i, box in enumerate(boxes):
        # box (x_left, y_top, x_right, y_bottom)
        #print(i)
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
        # 有些标注靠近边缘，无法收集足够满足要求的样本，因此此处用for循环，不用while
        for j in range(5000):
            size = npr.randint(int(max(w, h) * 1.1), np.ceil(1.6 * max(w, h)))

            # delta here is the offset of box center
            delta_x = npr.randint(-w * 0.2, w * 0.2)
            delta_y = npr.randint(-h * 0.2, h * 0.2)

            nx1 = int(max(x1 + w / 2 + delta_x - size / 2, 0))
            ny1 = int(max(y1 + h / 2 + delta_y - size / 2, 0)) # 转为int方便截图
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
            if IoU(crop_box, box_) >= 0.45 and cover(crop_box, box, 2):
                if p_idx < sample_per_box:
                    name = "%s_%d_%s" % (im_path, i, p_idx)
                    save_file = os.path.join(pos_save_dir, name + '.jpg')
                    f1.write(str(stdsize) + "/positive/" + name + ' 1 %f %f %f %f\n' % (
                    offset_x1, offset_y1, offset_x2, offset_y2))
                    cv2.imwrite(save_file, resized_im)
                    p_idx += 1
                    #print('p_idx %d' % p_idx)
                else:
                    p_total += p_idx # for循环在此处退出
                    d_total += d_idx
                    break
            elif IoU(crop_box, box_) >= 0.4:
                if d_idx < sample_per_box:
                    name = "%s_%d_%s" % (im_path, i, d_idx)
                    save_file = os.path.join(part_save_dir, name + '.jpg')
                    f3.write(str(stdsize) + "/part/" + name + ' -1 %f %f %f %f\n' % (
                    offset_x1, offset_y1, offset_x2, offset_y2))
                    cv2.imwrite(save_file, resized_im)
                    d_idx += 1
                    #print('d_idx %d' % d_idx)

    print("%s images done, pos: %s part: %s neg: %s"%(idx, p_total, d_total, n_total))

f1.close()
f2.close()
f3.close()
