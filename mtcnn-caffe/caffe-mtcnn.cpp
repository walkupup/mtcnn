

#include <cv.h>
#include <highgui.h>
#include "ccdl/include/classification.h"
#pragma comment(lib, "classification_dll.lib")

using namespace cv;

#define mydataFmt float
#define NumPoint   4
struct Bbox
{
	float score;
	float x1;
	float y1;
	float x2;
	float y2;
	float area;
	bool exist;
	mydataFmt ppoint[2 * NumPoint];
	mydataFmt regreCoord[4];

	operator Rect(){
		return Rect(x1, y1, x2-x1+1, y2-y1+1);
	}
};

struct orderScore
{
	mydataFmt score;
	int oriOrder;
};

bool cmpScore(struct orderScore lsh, struct orderScore rsh){
	if (lsh.score<rsh.score)
		return true;
	else
		return false;
}
void nms(vector<struct Bbox> &boundingBox_, vector<struct orderScore> &bboxScore_, const float overlap_threshold, string modelname){
	if (boundingBox_.empty()){
		return;
	}
	std::vector<int> heros;
	//sort the score
	sort(bboxScore_.begin(), bboxScore_.end(), cmpScore);

	int order = 0;
	float IOU = 0;
	float maxX = 0;
	float maxY = 0;
	float minX = 0;
	float minY = 0;
	while (bboxScore_.size()>0){
		order = bboxScore_.back().oriOrder;
		bboxScore_.pop_back();
		if (order<0)continue;
		heros.push_back(order);
		boundingBox_.at(order).exist = false;//delete it

		for (int num = 0; num<boundingBox_.size(); num++){
			if (boundingBox_.at(num).exist){
				//the iou
				maxX = (boundingBox_.at(num).x1>boundingBox_.at(order).x1) ? boundingBox_.at(num).x1 : boundingBox_.at(order).x1;
				maxY = (boundingBox_.at(num).y1>boundingBox_.at(order).y1) ? boundingBox_.at(num).y1 : boundingBox_.at(order).y1;
				minX = (boundingBox_.at(num).x2<boundingBox_.at(order).x2) ? boundingBox_.at(num).x2 : boundingBox_.at(order).x2;
				minY = (boundingBox_.at(num).y2<boundingBox_.at(order).y2) ? boundingBox_.at(num).y2 : boundingBox_.at(order).y2;
				//maxX1 and maxY1 reuse 
				maxX = ((minX - maxX + 1)>0) ? (minX - maxX + 1) : 0;
				maxY = ((minY - maxY + 1)>0) ? (minY - maxY + 1) : 0;
				//IOU reuse for the area of two bbox
				IOU = maxX * maxY;
				if (!modelname.compare("Union"))
					IOU = IOU / (boundingBox_.at(num).area + boundingBox_.at(order).area - IOU);
				else if (!modelname.compare("Min")){
					IOU = IOU / ((boundingBox_.at(num).area<boundingBox_.at(order).area) ? boundingBox_.at(num).area : boundingBox_.at(order).area);
				}
				if (IOU>overlap_threshold){
					boundingBox_.at(num).exist = false;
					for (vector<orderScore>::iterator it = bboxScore_.begin(); it != bboxScore_.end(); it++){
						if ((*it).oriOrder == num) {
							(*it).oriOrder = -1;
							break;
						}
					}
				}
			}
		}
	}
	for (int i = 0; i<heros.size(); i++)
		boundingBox_.at(heros.at(i)).exist = true;
}

vector<float> getScales(int w, int h, int minsize = 12){
	int row = h, col = w;
	float minl = row<col ? row : col;
	int MIN_DET_SIZE = 12;
	float m = (float)MIN_DET_SIZE / minsize;
	float factor = 0.709;
	int factor_count = 0;
	vector<float> scales_;

	while (minl*m>MIN_DET_SIZE){
		scales_.push_back(m);
		m *= factor;
	}
	return scales_;
}

#if 0
//这是一个简单的演示程序，用来看网络的结果是否正确
void main(){
	float means[] = { 127.5f, 127.5f, 127.5f };
	Classifier c("deploy-12pnet.prototxt", "12pnet-_iter_145542.caffemodel", 0.0078125, 0, 3, means, -1);
	Mat img = imread("300.jpg");
	//c.reshape(img.cols, img.rows);
	c.forward(img);
	WPtr<BlobData> cls = c.getBlobData("cls_loss");
	WPtr<BlobData> box = c.getBlobData("conv4-2");

	Mat cls_map(cls->height, cls->width, CV_32F, cls->list+cls->width * cls->height);
}
#endif

#if 0
//这是ONet的简单演示程序
void main(){
	float means[] = {127.5f,127.5f,127.5f};
	Classifier c("deploy-48onet.prototxt", "48onet-_iter_5726.caffemodel", 0.0078125, 0, 3, means, 0);
	Mat img = imread("300.jpg");
	Rect roi(623, 170, 109, 109);
	Mat img2;
	resize(img(roi), img2, Size(48, 48));
	
	c.forward(img2);
	WPtr<BlobData> cls = c.getBlobData("cls_loss");
	float conf = cls->list[1];
	WPtr<BlobData> box = c.getBlobData("conv6-2");
	float lx = box->list[0];
	float ly = box->list[1];
	float rx = box->list[2];
	float ry = box->list[3];

	float newlx = roi.x + lx*roi.width;
	float newly = roi.y + ly*roi.height;
	float newrx = roi.x + rx*roi.width;
	float newry = roi.y + ry*roi.height;
	rectangle(img, Point(newlx, newly), Point(newrx, newry), Scalar(0, 255), 2);
}
#endif

//由于我们的调试目录指向了ccdl/x64下，所以需要通过2级目录回到工程目录下
#define CDRoot "../../"

void main(){

	float means[] = { 127.5f, 127.5f, 127.5f };

	//如果使用GPU版本，最后的-1参数给0就好了
	Classifier pnet(CDRoot "models/det1.prototxt", CDRoot "models/det1.caffemodel", 0.0078125, 0, 3, means, -1);
	Classifier rnet(CDRoot "models/det2.prototxt", CDRoot "models/det2.caffemodel", 0.0078125, 0, 3, means, -1);
	Classifier onet(CDRoot "models/det3.prototxt", CDRoot "models/det3.caffemodel", 0.0078125, 0, 3, means, -1);

	//VideoCapture cap(0);
	double tick = 0;
	Mat img = imread(CDRoot "00284.jpg");
	Mat raw_img = img.clone();
	//cap >> raw_img;

	float confs[] = {0.5, 0.6, 0.8};
	float nmss[] = {0.2, 0.2, 0.2};
	vector<float> scales_ = getScales(raw_img.cols, raw_img.rows);

	//因为这里是为了调试视频，所以加了while，事实上这个while在图片下并没用，就执行一次
	while (!raw_img.empty()){
		vector<Bbox> pnetBoxAll;
		vector<orderScore> pnetOrderAll;

		tick = getTickCount();

		///////////////////////////////////////////////////////PNet
		int wnd = 12;
		for (int k = 0; k < scales_.size(); ++k){
			vector<Bbox> pnetBox;
			vector<orderScore> pnetOrder;
			float scale = scales_[k];
			resize(raw_img, img, Size(ceil(raw_img.cols*scale), ceil(raw_img.rows*scale)), 0, 0, cv::INTER_LINEAR);
			pnet.reshape(img.cols, img.rows);
			pnet.forward(img);

			//WPtr是包含了自动释放的功能，一个简单的智能指针实现
			WPtr<BlobData> cls_loss = pnet.getBlobData("prob1");
			Mat channels1(cls_loss->height, cls_loss->width, CV_32F, cls_loss->list + cls_loss->height*cls_loss->width);
			WPtr<BlobData> box = pnet.getBlobData("conv4-2");
			Mat loc(box->height, box->width, CV_32F, box->list);

			int stride = 2;
			for (int i = 0; i < channels1.cols; ++i){
				for (int j = 0; j < channels1.rows; ++j){
					float conf = channels1.at<float>(j, i);
					if (conf >= confs[0]){
						int cellx = i;
						int celly = j;
						float raw_x = (cellx * stride) / scale;
						float raw_y = (celly * stride) / scale;
						float raw_r = (cellx * stride + wnd) / scale;
						float raw_b = (celly * stride + wnd) / scale;

						int plane_size = box->width * box->height;
						float x1 = *(box->list + i + j * box->width + plane_size * 0);
						float y1 = *(box->list + i + j * box->width + plane_size * 1);
						float x2 = *(box->list + i + j * box->width + plane_size * 2);
						float y2 = *(box->list + i + j * box->width + plane_size * 3);

						float raw_w = raw_r - raw_x + 1;
						float raw_h = raw_b - raw_y + 1;
						x1 = raw_x + x1 * raw_w;
						y1 = raw_y + y1 * raw_h;
						x2 = raw_x + x2 * raw_w;
						y2 = raw_y + y2 * raw_h;

						raw_w = x2 - x1 + 1;
						raw_h = y2 - y1 + 1;

						Bbox b;
						b.area = raw_w * raw_h;
						b.exist = true;
						b.score = conf;
						b.x1 = x1;
						b.y1 = y1;
						b.x2 = x2;
						b.y2 = y2;
						pnetBox.push_back(b);

						orderScore order;
						order.oriOrder = pnetOrder.size();
						order.score = conf;
						pnetOrder.push_back(order);
					}
				}
			}

			nms(pnetBox, pnetOrder, 0.5, "Union");
			for (int i = 0; i < pnetBox.size(); ++i){
				if (pnetBox[i].exist){
					pnetBoxAll.push_back(pnetBox[i]);

					orderScore order;
					order.oriOrder = pnetOrderAll.size();
					order.score = pnetBox[i].score;
					pnetOrderAll.push_back(order);
				}
			}
		}
		nms(pnetBoxAll, pnetOrderAll, nmss[0], "Union");


		/////////////////////////////////////////////////////////////RNet
		vector<Bbox> rnetBoxAll;
		vector<orderScore> rnetOrderAll;
		for (int i = 0; i < pnetBoxAll.size(); ++i){
			if (pnetBoxAll[i].exist){
				Rect cropBox = pnetBoxAll[i];
				int size = (cropBox.width + cropBox.height) * 0.5;
				int cx = cropBox.x + cropBox.width * 0.5;
				int cy = cropBox.y + cropBox.height * 0.5;
				cropBox.x = cx - size*0.5;
				cropBox.y = cy - size*0.5;
				cropBox.width = size;
				cropBox.height = size;

				//cropBox的存在是为了保证输入给网络的是一个正方图，因为我们训练的时候也是这样的不是么
				cropBox = cropBox & Rect(0, 0, raw_img.cols, raw_img.rows);
				if (cropBox.width <= 0 && cropBox.height <= 0)
					continue;

				Mat img2;
				resize(raw_img(cropBox), img2, Size(24, 24));
				rnet.forward(img2);
				WPtr<BlobData> cls = rnet.getBlobData("prob1");
				float conf = cls->list[1];
				printf("RNet: %f > %f\n", conf, confs[1]);
				if (conf > confs[1])
				{
					WPtr<BlobData> box = rnet.getBlobData("conv5-2");
					float lx = box->list[0];
					float ly = box->list[1];
					float rx = box->list[2];
					float ry = box->list[3];

					float newlx = cropBox.x + lx*cropBox.width;
					float newly = cropBox.y + ly*cropBox.height;
					float newrx = cropBox.x + rx*cropBox.width;
					float newry = cropBox.y + ry*cropBox.height;
					//rectangle(raw_img, Point(newlx, newly), Point(newrx, newry), Scalar(0, 255), 2);

					orderScore order;
					order.oriOrder = rnetOrderAll.size();
					order.score = conf;
					rnetOrderAll.push_back(order);

					Bbox b;
					b = pnetBoxAll[i];
					b.x1 = newlx;
					b.y1 = newly;
					b.x2 = newrx;
					b.y2 = newry;
					b.score = conf;
					b.exist = true;
					b.area = (b.x2 - b.x1)*(b.y2 - b.y1);
					rnetBoxAll.push_back(b);
				}
			}
		}
		nms(rnetBoxAll, rnetOrderAll, nmss[1], "Union");


		/////////////////////////////////////////////////////////////ONet
		vector<Bbox> onetBoxAll;
		vector<orderScore> onetOrderAll;
		for (int i = 0; i < rnetBoxAll.size(); ++i){
			if (rnetBoxAll[i].exist){
				Rect cropBox = rnetBoxAll[i];
				int size = (cropBox.width + cropBox.height) * 0.5;
				int cx = cropBox.x + cropBox.width * 0.5;
				int cy = cropBox.y + cropBox.height * 0.5;
				cropBox.x = cx - size*0.5;
				cropBox.y = cy - size*0.5;
				cropBox.width = size;
				cropBox.height = size;

				cropBox = cropBox & Rect(0, 0, raw_img.cols, raw_img.rows);
				if (cropBox.width <= 0 && cropBox.height <= 0)
					continue;

				Mat img2;
				resize(raw_img(cropBox), img2, Size(48, 48));
				onet.forward(img2);
				WPtr<BlobData> cls = onet.getBlobData("prob1");
				float conf = cls->list[1];
				printf("ONet: %f > %f\n", conf, confs[2]);
				if (conf > confs[2])
				{
					WPtr<BlobData> box = onet.getBlobData("conv6-2");
					float lx = box->list[0];
					float ly = box->list[1];
					float rx = box->list[2];
					float ry = box->list[3];

					float newlx = cropBox.x + lx*cropBox.width;
					float newly = cropBox.y + ly*cropBox.height;
					float newrx = cropBox.x + rx*cropBox.width;
					float newry = cropBox.y + ry*cropBox.height;
					//rectangle(raw_img, Point(newlx, newly), Point(newrx, newry), Scalar(0, 255), 2);

					orderScore order;
					order.oriOrder = onetOrderAll.size();
					order.score = conf;
					onetOrderAll.push_back(order);

					//如果有坐标就使用他
					//WPtr<BlobData> pts = onet.getBlobData("conv6-3");
					Bbox b;
					b = rnetBoxAll[i];
					b.x1 = newlx;
					b.y1 = newly;
					b.x2 = newrx;
					b.y2 = newry;
					b.score = conf;
					b.exist = true;
					b.area = (b.x2 - b.x1)*(b.y2 - b.y1);

					/*
					for (int k = 0; k < NumPoint; ++k){
						int xp = k * 2;
						int yp = k * 2 + 1;
						float x = pts->list[xp] * cropBox.width + cropBox.x;
						float y = pts->list[yp] * cropBox.height + cropBox.y;
						b.ppoint[xp] = x;
						b.ppoint[yp] = y;
					}
					*/
					onetBoxAll.push_back(b);
				}
			}
		}
		nms(onetBoxAll, onetOrderAll, nmss[2], "Union");

		for (int i = 0; i < onetBoxAll.size(); ++i){
			if (onetBoxAll[i].exist){
				rectangle(raw_img, onetBoxAll[i], Scalar(0, 255), 2);
			}
		}
		
		tick = (getTickCount() - tick) / getTickFrequency() * 1000;
		printf("times: %.2f ms\n", tick);
		imshow("video-demo", raw_img);
		waitKey(0);

		break;
		//cap >> raw_img;
	}
}