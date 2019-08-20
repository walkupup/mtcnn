#include <fstream>
#include "fdetection.h"
#include <opencv2/opencv.hpp>

using namespace cv;

#define MAXFACEOPEN 0 //设置是否开关最大人脸调试，1为开，其它为关
/*
void test_video() {
	char *model_path = "../model";
	MTCNN mtcnn(model_path);
	mtcnn.SetMinFace(40);
	cv::VideoCapture mVideoCapture(0);
	if (!mVideoCapture.isOpened()) {
		return;
	}
	cv::Mat frame;
	mVideoCapture >> frame;
	while (!frame.empty()) {
		mVideoCapture >> frame;
		if (frame.empty()) {
			break;
		}

		clock_t start_time = clock();
		
		ncnn::Mat ncnn_img = ncnn::Mat::from_pixels(frame.data, ncnn::Mat::PIXEL_BGR2RGB, frame.cols, frame.rows);
		std::vector<Bbox> finalBbox, rawBbox;
#if(MAXFACEOPEN==1)
		mtcnn.detectMaxFace(ncnn_img, finalBbox);
#else
		mtcnn.detect(ncnn_img, finalBbox, rawBbox);
#endif
		const int num_box = finalBbox.size();
		std::vector<cv::Rect> bbox;
		bbox.resize(num_box);
		for(int i = 0; i < num_box; i++){
			bbox[i] = cv::Rect(finalBbox[i].x1, finalBbox[i].y1, finalBbox[i].x2 - finalBbox[i].x1 + 1, finalBbox[i].y2 - finalBbox[i].y1 + 1);
		
			for (int j = 0; j<5; j = j + 1)
			{
				cv::circle(frame, Point(finalBbox[i].ppoint[j], finalBbox[i].ppoint[j + 5]), 2, CV_RGB(0, 255, 0), FILLED);
			}
		}
		for (vector<cv::Rect>::iterator it = bbox.begin(); it != bbox.end(); it++) {
			rectangle(frame, (*it), Scalar(0, 0, 255), 2, 8, 0);
		}
		imshow("face_detection", frame);
		clock_t finish_time = clock();
		double total_time = (double)(finish_time - start_time) / CLOCKS_PER_SEC;
		std::cout << "time" << total_time * 1000 << "ms" << std::endl;
	
		int q = cv::waitKey(10);
		if (q == 27) {
			break;
		}
	}
	return ;
}
*/

int test_picture(FDetection *hdl, std::string filename){
	cv::Mat image;
	std::vector<cv::Rect> box;

	image = cv::imread(filename);
	if (image.empty())
	{
		return -1;
	}
	clock_t start_time = clock();
	hdl->detect(image, box);
	clock_t finish_time = clock();
	double total_time = (double)(finish_time - start_time) / CLOCKS_PER_SEC;

	const int num_box = box.size();
	for (std::vector<cv::Rect>::iterator it = box.begin(); it != box.end(); it++) {
		rectangle(image, (*it), Scalar(0, 0, 255), 2, 8, 0);
		//circle(image, cv::Point((*it).x + (*it).width / 2, (*it).y + (*it).height / 2), 20, cv::Scalar(0, 0, 255), 2);
	}
	//for (int i = 0; i < rawBbox.size(); i++) {
	//	cv::Rect b = cv::Rect(rawBbox[i].x1, rawBbox[i].y1, rawBbox[i].x2 - rawBbox[i].x1 + 1, rawBbox[i].y2 - rawBbox[i].y1 + 1);
	//	rectangle(image, b, Scalar(255, 0, 255), 1, 8, 0);
	//}

	imshow("face_detection", image);
	std::cout << "time" << total_time * 1000 << "ms" << std::endl;

	cv::waitKey();
	return 1;
}

void test_video()
{
	FDetection hdl;
	hdl.init();
	cv::VideoCapture vid("F:\\data\\smartcar\\20190726\\chess.avi");
	cv::VideoWriter vw("output.avi", VideoWriter::fourcc('M', 'J', 'P', 'G'), 5, cv::Size((int)vid.get(cv::CAP_PROP_FRAME_WIDTH), (int)vid.get(cv::CAP_PROP_FRAME_HEIGHT)), true);
	cv::Mat frame;
	int frameNum = 0;
	while (1)
	{
		std::vector<cv::Rect> box;
		vid >> frame;
		if (frame.empty())
		{
			break;
		}
		if (frameNum++ % 25 != 0)
			continue;
		hdl.detect(frame, box);
		const int num_box = box.size();
		for (std::vector<cv::Rect>::iterator it = box.begin(); it != box.end(); it++) {
			rectangle(frame, (*it), Scalar(0, 0, 255), 2, 8, 0);
			//circle(image, cv::Point((*it).x + (*it).width / 2, (*it).y + (*it).height / 2), 20, cv::Scalar(0, 0, 255), 2);
		}
		//for (int i = 0; i < rawBbox.size(); i++) {
		//	cv::Rect b = cv::Rect(rawBbox[i].x1, rawBbox[i].y1, rawBbox[i].x2 - rawBbox[i].x1 + 1, rawBbox[i].y2 - rawBbox[i].y1 + 1);
		//	rectangle(image, b, Scalar(255, 0, 255), 1, 8, 0);
		//}

		imshow("face_detection", frame);
		vw << frame;
		//std::cout << "time" << total_time * 1000 << "ms" << std::endl;
		char c = cv::waitKey(1);
		if (c == 'c')
			break;
	}
	hdl.release();
	vw.release();
}

int main(int argc, char** argv) {
	//test_video();
	//return 1;
	std::string listFile = "D:\\data\\mine\\list1-70.txt";
	std::ifstream in(listFile);
	std::string line;
	std::vector<std::string> names;
	while (in >> line)
	{
		names.push_back(line);
	}
	
	FDetection hdl;
	hdl.init();

	for (int i = 0; i < names.size(); i++)
		test_picture(&hdl, names[i]);
	hdl.release();

	return 0;
}