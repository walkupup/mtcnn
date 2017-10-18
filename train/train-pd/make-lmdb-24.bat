@echo off

if exist train_lmdb24 rd /q /s train_lmdb24
if exist val_lmdb24 rd /q /s val_lmdb24

echo create train_lmdb24...
"caffe/convert_imageset.exe" "" 24/label-train.txt train_lmdb24 --backend=mtcnn --shuffle=true

echo done.

echo create val_lmdb24...
"caffe/convert_imageset.exe" "" 24/label-val.txt val_lmdb24 --backend=mtcnn --shuffle=true

echo done.


pause