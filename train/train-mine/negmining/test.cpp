#include<iostream>
#include<vector>
//#include"net.h"
//#include"mat.h"
#include<opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <fstream>
#include <cstdlib>
#include<string>
using namespace cv;
using namespace cv::dnn;
using namespace std;
//void test() {
//	ncnn::Net net;
//	net.load_param("detone.param");//  .param
//	net.load_model("detone.bin");
//	cv::Mat image;
//	ifstream in("E:/xingren/test/resize_pos/file.txt");
//	std::string name;
//	std::vector<string> vname;
//	while (in >> name) {
//		vname.push_back(name);
//	}
//	int tp = 0;
//	for (int i = 0; i < vname.size(); i++) {
//		Mat img = cv::imread("E:/xingren/test/" + vname[i]);
//		imshow("a", img);
//		ncnn::Mat ncnn_img = ncnn::Mat::from_pixels(img.data, ncnn::Mat::PIXEL_BGR2RGB, img.cols, img.rows);
//		ncnn::Mat in;
//		ncnn::Mat out;
//		resize_bilinear(ncnn_img, in, 8, 24);
//		const float mean_vals[3] = { 127.5f, 127.5f, 127.5f };
//		const float norm_vals[3] = { 0.0078125, 0.0078125, 0.0078125 };
//		in.substract_mean_normalize(mean_vals, norm_vals);
//		ncnn::Extractor ex = net.create_extractor();
//		ex.set_light_mode(true);
//		ex.input("data", in);
//		ex.extract("prob1", out);
//		cout << "out[0]="<<out[0] << endl;
//		if (out[0] > 0.5) {
//			float x = tp++;
//			cout << "x= " << x << endl;
//		}
//	/*	cout << float(out[1]) << endl;*/
//	}
//	cout <<"size="<< vname.size() << endl;
//	float score = float(tp) / vname.size();
//	cout << "score"<< ", "<<score << endl;
//	net.clear();
//	cv::waitKey();
//}

// 截取cropWidth，cropHeight的窗口，resize成stdWidth, stdHeight，放到vImgs里面
void generateBoxes(cv::Mat image, int cropWidth, int cropHeight, int stdWidth, int stdHeight, std::vector<cv::Mat> &vImgs)
{
	int width = image.cols;
	int height = image.rows;
	vImgs.clear();
	int step = cropWidth / 4;
	for (int i = 0; i < height - cropHeight; i += step)
	{
		for (int j = 0; j < width - cropWidth; j += step)
		{
			cv::Rect roi(j, i, cropWidth, cropHeight);
			cv::Mat simg;
			cv::resize(image(roi), simg, cv::Size(stdWidth, stdHeight));
			vImgs.push_back(simg);
		}
	}
}

std::string getFileName(std::string file)
{
	
	int pos = file.rfind("/");
	//cout << file << " " << pos << endl;
	//cout << file.substr(pos + 1, file.length() - pos - 1) << endl;
	return file.substr(pos + 1, file.length() - pos - 1);
}

void opencv_test(char *rootDir, char *fileList) 
{
	vector<Mat> img;
	String modelTxt = "../model/det1.prototxt";
	String modelBin = "../model/_iter_20000.caffemodel";
	//String modelTxt = "./zifu/lenet_deploy.prototxt";
	//String modelBin = "./zifu/_iter_10000.caffemodel";
	Net net = dnn::readNetFromCaffe(modelTxt, modelBin);
	//Net net = readNet(model, config, framework);
    net.setPreferableBackend(0);
    net.setPreferableTarget(0);
	int boxW = 20;
	int boxH = 20;
	int cropWidth, cropHeight;
	float scale = 1.2f;
	ifstream in(fileList);
	string name;
	vector <string> vname;
	int total = 0;
	char pic[10000];
	while (in >> name)
	{
		vname.push_back(name);
	}
	for (int i = 0; i < vname.size(); i++) {
		//Mat img = cv::imread("E:/xingren/test/" + vname[i]);
		std::string imageName = rootDir + vname[i];
		std::cout << "processing: " << i << " " << imageName << std::endl;
		Mat image = cv::imread(imageName);
		if (image.empty())
		{
			std::cout << "Can't read: " << imageName << std::endl;
			continue;
		}
		//imshow("a", image);
		//cv::waitKey(1);
		cropWidth = boxW;
		cropHeight = boxH;
		int fp = 0;
		while (cropWidth < image.cols && cropHeight < image.rows)
		{
			std::vector<cv::Mat> vImgs;
			generateBoxes(image, cropWidth, cropHeight, boxW, boxH, vImgs);
			for (size_t j = 0; j < vImgs.size(); j++)
			{
				Mat inputBlob = blobFromImage(vImgs[j], 0.0078125f, Size(boxW, boxH), Scalar(127.5, 127.5, 127.5), false);
				net.setInput(inputBlob, "data");
				Mat prob = net.forward("prob1");
				float p = ((float *)prob.data)[1];
				//cout << prob.ptr<float>(0)[1] << endl;
				//cout << "score= " << p << endl;
				if (p > 0.5) 
				{
					sprintf(pic, "./hardneg/%s_%05d.jpg", getFileName(vname[i]).c_str(), fp);
					imwrite(pic, vImgs[j]);
					fp++;
					total++;
				}
			}
			cropWidth = int(cropWidth * scale);
			cropHeight = int(cropHeight * scale);
		}
		std::cout << "neg: " << fp  << " total: " << total << std::endl;	
	}
}

int main(int argc, char **argv) {
	//test();
	if (argc != 3)
	{
		cout << "parameter error" << endl;
		return -1;
	}
	
	opencv_test(argv[1], argv[2]);
	return 0;
}
