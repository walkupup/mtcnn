#include "network.h"
#include "mtcnn.h"
#include <time.h>
#pragma comment(lib, "libopenblas.dll.a")
#include <fstream>
int main()
{
	//��Ϊִ��Ŀ¼�����õ�openblas/x64���ˣ���֤dll���������룬��ʱ��ͼƬ·�������Ҫ����ȥ2��
	ifstream listfile("pictureNameList.txt");
	std::string name;
	vector<std::string> names;
	std::string imageDir("F:\\data\\pd\\pd-pos\\");
	while (listfile >> name)
	{
		names.push_back(name);
	}
	for (int i = 0; i < names.size(); i++)
	{
		//Mat im_ = imread(imageDir + names[i]);
		Mat im_ = imread("p2.jpg");
		if (im_.empty())
			continue;
		Mat im = im_;
		//cv::resize(im_, im, cv::Size(im_.cols, im_.rows * 0.4));
		mtcnn find(im.cols, im.rows, 100);
		vector<Rect> objs = find.detectObject(im);
		for (int i = 0; i < objs.size(); ++i)
			rectangle(im, objs[i], Scalar(0, 255), 2);

		imshow("demo", im);
		waitKey(1);
	}
    return 0;
}