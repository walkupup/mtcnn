# 训练部分
下载：<br/>
舌头检测测试样本[train-samples-st.rar](www.zifuture.com/fs/12.github/mtcnn/train-samples-st.rar)<br/>


<br/>
## 无关键点的步骤：<br/>
1.准备好训练的样本图片放到samples文件夹<br/>
2.准备好对应的label.txt，格式是<br/>
   samples/filename.jpg xmin ymin xmax ymax<br/>
3.执行callpy-gen-data12.bat生成样本数据<br/>
4.执行callpy-12.bat生成训练需要用到的train-label.txt<br/>
5.执行make-lmdb-12.bat生成lmdb数据库<br/>
6.执行train-12.bat开始训练<br/>
7.相对应的其他24、48网络也类似就好了<br/>

<br/>
对于训练、转换数据集用到的程序，全在https://github.com/dlunion/CCDL/tree/master/caffe-easy这个版本的caffe里面，该caffe主要运行在windows下，可以复制里面主要的层和程序也可以完成任务<br/>

<br/>
## 有关键点的训练
产生一个label.txt的时候，格式是:<br/>
   samples/filename.jpg xmin ymin xmax ymax ptsx ptsy ptsx ptsy<br/>
然后相对应的修改py代码(gen_24net_list.py、gen_48net_list.py)里面的has_pts为True，注意里面-1的个数要跟你的pts个数对上（尴尬没写傻瓜式一点）<br/>

<br/>
## 一个图多个box的训练
主要是box的处理在gen_12net_data2.py的bbox = map(float, annotation[1:5])部分，这里限制了只读取1个box，如果多个box可以修改5为-1，当然这时候如果你又有pts就得自己修改啦。<br/>
