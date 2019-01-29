1. 运行 gen_net_data.py 生成训练样本
2. 运行 gen_net_list.py 生成训练样本列表label-train.txt，label-val.txt
3. 把 make-lmdb-18.bat 和 train-18.bat，caffe目录拷到训练图像目录下
4. 运行两个bat，训练。
5. 把训练好的caffemodel文件拷回到model目录下。
6. negmining 目录下负样本挖掘
7. 把负样本生成列表，加入到label-train.txt 中。
8. 重复4-7步，直到挖掘不出负样本。