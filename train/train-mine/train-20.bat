@echo off
"caffe/caffe.exe" train --solver=solver-18.prototxt --weights=./models-18/_iter_50000.caffemodel
pause