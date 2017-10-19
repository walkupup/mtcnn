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
#pragma comment(lib, "libopenblas.dll.a")
#include <fstream>



int main()
{
	// ‰»Î ”∆µ
	int num = 1;
	string  name;
	ifstream infile("E:\\mtcnn\\mtcnn-master\\mtcnn-master\\Single_Stage\\Video_name.txt");
	string videoDir("E:\\mtcnn\\mtcnn-master\\mtcnn-master\\Single_Stage\\Video\\");
	vector<string> nameList;
	while (infile >> name)
		nameList.push_back(name);
	for (int i = 0; i < nameList.size(); i++)
	{
		string video_path = videoDir + nameList[i];
		VideoCapture capture(video_path);
		if (!capture.isOpened())
			return -1;
		Mat frame,im;
		int count = 0;
		while (1)
		{
			capture >> frame;
			if (frame.empty())
				break;
			if (count % 10 == 0)
			{
				imshow("fafa", frame);
				waitKey(1);
				cv::resize(frame, im, cv::Size(frame.cols, frame.rows * 0.36));
				mtcnn find(im.cols, im.rows, 100);
				vector<Rect> objs = find.mining(im);
				for (int i = 0; i < objs.size(); ++i) 
				{
					//rectangle(im, objs[i], Scalar(0, 255), 2);
					Mat roi_img = im(Range(objs[i].y, objs[i].y + objs[i].height), Range(objs[i].x, objs[i].x + objs[i].width));
					std::string name = std::to_string(num) + ".jpg";
					std::string path = "save/" + name;
					cv::imwrite(path, roi_img);
					num++;
				}
			}
			count++;
		}

	}
}
