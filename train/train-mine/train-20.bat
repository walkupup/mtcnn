@echo off
"caffe/caffe.exe" train --solver=solver-20.prototxt --weights=./models-20/_iter_5000.caffemodel
pause