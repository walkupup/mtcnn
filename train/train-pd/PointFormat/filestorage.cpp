#include "stdio.h"

#include "opencv2/opencv.hpp"
#include <iostream>
#include <fstream>
using namespace cv;
using namespace std;

std::string labelFileName(string imageName)
{
	string name = imageName.substr(0, imageName.rfind('.'));
	name.append("_pose.xml");
	return name;
}

int main11()
{
	std::string imageDir("E:/mtcnn/mtcnn-master/mtcnn-master/pd-sample/train_sample/1772/samples/");
	std::string labelDir(imageDir + "label");
	std::string fileList(imageDir + "filename.txt");
	ofstream outfile(imageDir + "pd-points.txt");

	ifstream infile(fileList);
	std::string filename;
	vector<std::string> vfile;
	while (infile >> filename)
	{
		vfile.push_back(filename);
	}

	for (size_t i = 0; i < vfile.size(); i++)
	{
		std::string xmlname = labelFileName(vfile[i]);
		
		std::string lableFile = labelDir + "/" + xmlname;
		FileStorage fs(lableFile, FileStorage::READ);

		if (!fs.isOpened())
		{
			continue;
		}

		Mat data;
		fs["pose_0"] >> data;
		std::cout << vfile[i] << " " << data.size[0] << " " << data.size[1] << " " << data.size[2] << std::endl;
		if (data.size[0] == 0 || data.size[1] == 0 || data.size[2] == 0)
			continue;

		Mat img = imread(imageDir + "/" + vfile[i]);
		vector<vector<Point>> pts;
		for (int i = 0; i < data.size[0]; i++)
		{
			vector<Point> person;
			int discard = 0;
			for (int j = 0; j < data.size[1]; j++)
			{
				//if (data.at<float>(i, j, 0) == 0 && data.at<float>(i, j, 1) == 0)
				//{
				//	discard = 1;
				//	break;
				//}
				//else
					person.push_back(Point((int)(data.at<float>(i, j, 0) + 0.5f), (int)(data.at<float>(i, j, 1) + 0.5f)));
			}
			//if (!discard)
				pts.push_back(person);
		}
		
		if (pts.size() > 0)
		{
			outfile << vfile[i] << " " << pts.size();
			for (int i = 0; i < pts.size(); i++) // ÈËÊý.
			{
				for (size_t k = 0; k < pts[i].size(); k++)
				{
					circle(img, pts[i][k], 4, Scalar(0, 0, 255), -1);
					outfile << " " << pts[i][k].x << " " << pts[i][k].y;
				}
			}
			outfile << endl;
		}
		imshow("image", img);
		waitKey(10);
		fs.release();
	}



	return 1;
}