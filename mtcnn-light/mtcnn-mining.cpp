#include "mtcnn.h"

#ifdef modelFromHFile
#include "mtcnn_models.h"
#endif

vector<Rect> mtcnn::mining(Mat &image, vector<vector<Point2f>>& keys){
	keys.clear();

	vector<Rect> objs;
	struct orderScore order;
	int count = 0;
	for (size_t i = 0; i < scales_.size(); i++) {
		int changedH = (int)ceil(image.rows*scales_.at(i));
		int changedW = (int)ceil(image.cols*scales_.at(i));
		resize(image, reImage, Size(changedW, changedH), 0, 0, cv::INTER_LINEAR);
		simpleFace_[i].run(reImage, scales_.at(i));
		nms(simpleFace_[i].boundingBox_, simpleFace_[i].bboxScore_, simpleFace_[i].nms_threshold);

		for (vector<struct Bbox>::iterator it = simpleFace_[i].boundingBox_.begin(); it != simpleFace_[i].boundingBox_.end(); it++){
			if ((*it).exist){
				firstBbox_.push_back(*it);
				order.score = (*it).score;
				order.oriOrder = count;
				firstOrderScore_.push_back(order);
				count++;
			}
		}
		simpleFace_[i].bboxScore_.clear();
		simpleFace_[i].boundingBox_.clear();
	}
	//the first stage's nms
	if (count<1)return objs;
	nms(firstBbox_, firstOrderScore_, nms_threshold[0]);
	refineAndSquareBbox(firstBbox_, image.rows, image.cols, true);

	//for (vector<struct Bbox>::iterator it = firstBbox_.begin(); it != firstBbox_.end(); it++) {
	//	if ((*it).exist) {
	//		objs.push_back(Rect(it->x1, it->y1, it->x2 - it->x1 + 1, it->y2 - it->y1 + 1));
	//	}
	//}

	//second stage
	count = 0;
	for (vector<struct Bbox>::iterator it = firstBbox_.begin(); it != firstBbox_.end(); it++){
		if ((*it).exist){
			Rect temp((*it).x1, (*it).y1, (*it).x2 - (*it).x1, (*it).y2 - (*it).y1);
			if (temp.width < 12 || temp.height < 12){
				(*it).exist = false;
				continue;
			}

			Mat secImage;
			resize(image(temp), secImage, Size(24, 24), 0, 0, cv::INTER_LINEAR);
			refineNet.run(secImage);
			//printf("RNet: %f / %f\n", *(refineNet.score_->pdata + 1), refineNet.Rthreshold);
			if (*(refineNet.score_->pdata + 1)>refineNet.Rthreshold){
				memcpy(it->regreCoord, refineNet.location_->pdata, 4 * sizeof(mydataFmt));
				it->area = (it->x2 - it->x1)*(it->y2 - it->y1);
				it->score = *(refineNet.score_->pdata + 1);
				secondBbox_.push_back(*it);
				order.score = it->score;
				order.oriOrder = count++;
				secondBboxScore_.push_back(order);
			}
			else{
				(*it).exist = false;
			}
		}
	}
	if (count<1)return objs;
	nms(secondBbox_, secondBboxScore_, nms_threshold[1]);
	refineAndSquareBbox(secondBbox_, image.rows, image.cols, true);


	for (vector<struct Bbox>::iterator it = secondBbox_.begin(); it != secondBbox_.end(); it++) {
		if ((*it).exist) {
			objs.push_back(Rect(it->x1, it->y1, it->x2 - it->x1 + 1, it->y2 - it->y1 + 1));
		}
	}

	firstBbox_.clear();
	firstOrderScore_.clear();
	secondBbox_.clear();
	secondBboxScore_.clear();
	//thirdBbox_.clear();
	//thirdBboxScore_.clear();
	return objs;
}