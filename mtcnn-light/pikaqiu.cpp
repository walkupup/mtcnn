#include "network.h"
#include "mtcnn.h"
#include <time.h>
//#pragma comment(lib, "libopenblas.a")
#include <fstream>
int main()
{
	//��Ϊִ��Ŀ¼�����õ�openblas/x64���ˣ���֤dll���������룬��ʱ��ͼƬ·�������Ҫ����ȥ2��
	ifstream listfile("G:\\data\\pd\\pd-positive\\pictureNameList.txt");
	std::string imageDir("G:\\data\\pd\\pd-positive\\");
	std::string name;
	vector<std::string> names;
	while (listfile >> name)
	{
		names.push_back(name);
	}
	for (int i = 0; i < names.size(); i++)
	{
		Mat im_ = imread(imageDir + names[i]);
		//Mat im_ = imread("p2.jpg");
		if (im_.empty())
			continue;
		//Mat im = im_;
		Mat im;
		cv::resize(im_, im, cv::Size(im_.cols, im_.rows * 0.36));
		mtcnn find(im.cols, im.rows, 100);
		vector<Rect> objs = find.detectObject(im);
		for (int i = 0; i < objs.size(); ++i)
			rectangle(im, objs[i], Scalar(0, 255), 2);

		imshow("demo", im);
		waitKey(50);
	}
    return 0;
}