@echo off

set img_dir=.

if exist %img_dir%/train_lmdb18 rd /q /s %img_dir%/train_lmdb18
if exist %img_dir%/val_lmdb18 rd /q /s %img_dir%/val_lmdb18

echo create train_lmdb18...
"caffe/convert_imageset.exe" "" %img_dir%/18/label-train.txt train_lmdb18 --backend=mtcnn --shuffle=true

echo done.

echo create val_lmdb18...
"caffe/convert_imageset.exe" "" %img_dir%/18/label-val.txt val_lmdb18 --backend=mtcnn --shuffle=true

echo done.


pause