@echo off
"caffe/caffe.exe" train --solver=solver-48.prototxt --weights=det3.caffemodel
pause