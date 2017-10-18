#include "stdio.h"
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include "opencv2/opencv.hpp"
using namespace std;


//cv::Rect points2Rect(std::vector<cv::Point> pt, int maxWidth, int maxHeight)
//{
//	int top = pt[0].y;
//	int bottom = std::max(pt[10].y, pt[13].y);
//	int height = bottom - top;
//	int middle = ((pt[11].x + pt[8].x) / 2 + pt[1].x) / 2;
//	int width;
//	top -= height * 0.06;
//	if (top < 0)
//		top = 0;
//	bottom += height * 0.15;
//	if (bottom > maxHeight - 1)
//		bottom = maxHeight - 1;
//	cv::Rect rec;
//	rec.y = top;
//	rec.height = bottom - top +  1;
//	width = (bottom - top + 1) / 2;
//	width = (width >> 1) << 1; // 偶数.
//	rec.x = middle - width / 2;
//	rec.width = width;
//	return rec;
//}

cv::Rect points2Rect(std::vector<cv::Point> pt, int maxWidth, int maxHeight)
{
	int top = pt[0].y;
	int bottom = std::max(pt[10].y, pt[13].y);
	int height = bottom - top;
	int left = pt[0].x;
	int right = pt[0].x;
	for (size_t i = 0; i < pt.size(); i++)
	{
		if (pt[i].x > right)
			right = pt[i].x;
		if (pt[i].x < left)
			left = pt[i].x;
	}
	//top -= height * 0.06;
	if (top < 0)
		top = 0;
	//bottom += height * 0.15;
	if (bottom > maxHeight - 1)
		bottom = maxHeight - 1;
	if (left < 0)
		left = 0;
	if (right > maxWidth - 1)
		right = maxWidth - 1;
	cv::Rect rec;
	rec.y = top;
	rec.height = bottom - top + 1;
	rec.x = left;
	rec.width = right - left + 1;
	return rec;
}

int main()
{
	std::string imageDir("E:/mtcnn/mtcnn-master/mtcnn-master/pd-sample/train_sample/1772/samples/");
	std::string imageOutDir("E:/mtcnn/mtcnn-master/mtcnn-master/pd-sample/train_sample/1772/samples0.4/");
	//std::string imageDir("E:/data/pd/pd-positive/");// ("C:/data/pd/pd-positive/");
	std::string fileName(imageDir + "pd-points.txt");
	std::string rectFileName(imageDir + "pd-rect.txt");
	int numPoints = 15;

	// 读取所有记录.
	std::ifstream infile(fileName);
	std::ofstream outfile(rectFileName);
	std::string buf, name;
	int n;
	while (getline(infile, buf))
	{
		istringstream ss;
		ss.str(buf);
		ss >> name;
		ss >> n;
		cv::Mat img = cv::imread(imageDir + "/" + name);
		if (img.empty())
			continue;
		outfile << name;
		cv::Mat img1;
		cv::resize(img, img1, cv::Size(img.cols, img.rows * 0.4));
		cv::imwrite(imageOutDir + name, img1);
		for (int i = 0; i < n; i++)
		{
			std::vector<cv::Point> pt(numPoints);
			for (int j = 0; j < numPoints; j++)
			{
				ss >> pt[j].x >> pt[j].y;
			}
			cv::Rect rec = points2Rect(pt, img.cols, img.rows);
			outfile << " " << rec.x << " " << int(rec.y * 0.4) << " " << rec.x + rec.width - 1 << " " << int((rec.y + rec.height - 1) * 0.4);
			rectangle(img, rec, cv::Scalar(0, 0, 255), 1);
		}
		outfile << endl;
		imshow("img", img);
		cv::waitKey(1);
	}

	return 1;
}