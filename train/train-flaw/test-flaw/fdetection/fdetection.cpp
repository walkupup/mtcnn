
#include "fdetection.h"
#include "mtcnn.h"
#include "opencv2/opencv.hpp"

void FDetection::init()
{
	char *model_path = "../../model-smartcar";
	det = new MTCNN(model_path);
}
void FDetection::release()
{
	delete(det);
}

void FDetection::detect(cv::Mat image, std::vector<cv::Rect> &box)
{
	MTCNN *mtcnn = (MTCNN *)det;
	std::vector<Bbox> finalBbox, rawBbox;
	ncnn::Mat ncnn_img = ncnn::Mat::from_pixels(image.data, ncnn::Mat::PIXEL_BGR, image.cols, image.rows);

	mtcnn->detect(ncnn_img, finalBbox, rawBbox);
	box.clear();
	for (int i = 0; i < finalBbox.size(); i++)
	{
		box.push_back(cv::Rect(finalBbox[i].x1, finalBbox[i].y1, finalBbox[i].x2 - finalBbox[i].x1 + 1, finalBbox[i].y2 - finalBbox[i].y1 + 1));
	}
	return;
}

