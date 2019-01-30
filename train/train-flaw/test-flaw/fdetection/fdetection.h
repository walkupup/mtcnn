#pragma once
#include "opencv2/opencv.hpp"

class FDetection
{
public:
	void *det;

	void init();
	void release();

	void FDetection::detect(cv::Mat image, std::vector<cv::Rect> &box);

};
