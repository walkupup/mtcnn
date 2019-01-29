#include "mtcnn.h"
#include <opencv2/opencv.hpp>

using namespace cv;

#define MAXFACEOPEN 0 //设置是否开关最大人脸调试，1为开，其它为关

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
		std::vector<Bbox> finalBbox;
#if(MAXFACEOPEN==1)
		mtcnn.detectMaxFace(ncnn_img, finalBbox);
#else
		mtcnn.detect(ncnn_img, finalBbox);
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

void show_ncnnMat(ncnn::Mat img, std::string name, float shift, float scale)
{
	cv::Mat img1(cv::Size(img.w, img.h), CV_32FC1, img.channel(0));
	cv::Mat img_show;
	cv::Mat img_;
	img1.copyTo(img_);
	img_ = img_ * scale;
	img_ = img_ + shift;
	img_.convertTo(img_show, CV_8UC1);
	cv::imwrite(name + ".bmp", img_show);
	cv::imshow(name, img_show);
	//cv::waitKey();
}

int test_picture(std::string filename){
	char *model_path = "../../model";
	MTCNN mtcnn(model_path);
	//mtcnn.testNet();
	//return 1;

	cv::Mat image;
	image = cv::imread(filename);
	clock_t start_time = clock();
	ncnn::Mat ncnn_img = ncnn::Mat::from_pixels(image.data, ncnn::Mat::PIXEL_BGR, image.cols, image.rows);
	std::vector<Bbox> finalBbox;

#if(MAXFACEOPEN==1)
	mtcnn.detectMaxFace(ncnn_img, finalBbox);
#else
	mtcnn.detect(ncnn_img, finalBbox);
#endif
	clock_t finish_time = clock();
	double total_time = (double)(finish_time - start_time) / CLOCKS_PER_SEC;

	const int num_box = finalBbox.size();
	std::vector<cv::Rect> bbox;
	bbox.resize(num_box);
	for (int i = 0; i < num_box; i++) {
		bbox[i] = cv::Rect(finalBbox[i].x1, finalBbox[i].y1, finalBbox[i].x2 - finalBbox[i].x1 + 1, finalBbox[i].y2 - finalBbox[i].y1 + 1);

		//for (int j = 0; j<5; j = j + 1)
		//{
		//	cv::circle(image, Point(finalBbox[i].ppoint[j], finalBbox[i].ppoint[j + 5]), 2, CV_RGB(0, 255, 0), FILLED);
		//}
	}
	for (vector<cv::Rect>::iterator it = bbox.begin(); it != bbox.end(); it++) {
		rectangle(image, (*it), Scalar(0, 0, 255), 2, 8, 0);
	}

	imshow("face_detection", image);
	std::cout << "time" << total_time * 1000 << "ms" << std::endl;

	cv::waitKey(0);
	return 1;
}

int main(int argc, char** argv) {
	//test_video();
	std::string files[8] = {"D:/data/slw/1/1.png", "D:/data/slw/1/2.png", "D:/data/slw/1/3.png"
		, "D:/data/slw/1/4.png" , "D:/data/slw/1/5.png" , "D:/data/slw/1/6.png" , "D:/data/slw/1/7.png" 
		, "D:/data/slw/1/8.png"};
	for (int i = 0; i < 8; i++)
		test_picture(files[i]);
	return 0;
}