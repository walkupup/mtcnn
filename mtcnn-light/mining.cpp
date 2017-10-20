#include "network.h"
#include "mtcnn.h"
#include <time.h>
#include <opencv2/core/core.hpp>  
#include <opencv2/highgui/highgui.hpp>  
#include <opencv2/imgproc/imgproc.hpp>  
#include <opencv2/videoio.hpp>
#include <opencv2/video/video.hpp>
#include <iostream> 
#include <iomanip> 
#include <stdlib.h>
//#pragma comment(lib, "libopenblas.dll.a")
#include <fstream>

cv::Rect rect2Square(cv::Rect it)
{
	int nw = it.width;
	int nh = it.height;
	//float size = (nw + nh) * 0.5;
	float size = std::max(nw, nh) * (1.3f + 0.85f) * 0.5f;
	float cx = it.x + nw * 0.5;
	float cy = it.y + nh * 0.5;
	cv::Rect rec;
	rec.x = cx - size * 0.5;
	rec.y = cy - size * 0.5;
	rec.width = size;
	rec.height = size;
	return rec;
}

int main()
{
	//������Ƶ
	string  name;
	//string videoDir("H:\\movie\\");
	string videoDir("E:\\Video\\");
	ifstream infile(videoDir + "video.txt");
	vector<string> nameList;
	while (infile >> name)
		nameList.push_back(name);
	for (int j = 0; j < nameList.size(); j++)
	{
		int num = 1;
		string video_path = videoDir + nameList[j]+".mp4";
		VideoCapture capture(video_path);
		if (!capture.isOpened())
			continue;
		Mat frame, im;
		int count = 0;
		while (1)
		{
			capture >> frame;
			if (frame.empty())
				break;
			if (count % 30 == 0)
			{
				//imshow("fafa", frame);
				//waitKey(1);
				cout << nameList[j] << " " << count << endl;
				cv::resize(frame, im, cv::Size(frame.cols, frame.rows *1));
				mtcnn find(im.cols, im.rows, 48);
				vector<Rect> objs = find.mining(im);
				for (int i = 0; i < objs.size(); ++i) 
				{
					cv::Rect rec = rect2Square(objs[i]);
					if (rec.x < 0 || rec.x + rec.width > im.cols || rec.y < 0 || rec.y + rec.height > im.rows)
						continue;
					cv::Mat img24, img48;
					cv::resize(im(rec), img24, cv::Size(24, 24));
					cv::resize(im(rec), img48, cv::Size(48, 48));
   				    std::string name = nameList[j] + "_" + std::to_string(num) + ".jpg";
//					std::string name = std::to_string(num) + ".jpg";
					cv::imwrite("24/" + name, img24);
					cv::imwrite("48/" + name, img48);
					num++;
				}
			}
			count++;
		}

	}
}
