@echo off

set img_dir=D:/data/mine

if exist %img_dir%/train_lmdb20 rd /q /s %img_dir%/train_lmdb20
if exist %img_dir%/val_lmdb20 rd /q /s %img_dir%/val_lmdb20

echo create train_lmdb20...
"caffe/convert_imageset.exe" "" %img_dir%/20/label-train.txt train_lmdb20 --backend=mtcnn --shuffle=true

echo done.

echo create val_lmdb18...
"caffe/convert_imageset.exe" "" %img_dir%/20/label-val.txt val_lmdb20 --backend=mtcnn --shuffle=true

echo done.


pause