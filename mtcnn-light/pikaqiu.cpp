#include "network.h"
#include "mtcnn.h"
#include <time.h>
#pragma comment(lib, "libopenblas.dll.a")

int main()
{
	//因为执行目录被设置到openblas/x64下了，保证dll能正常载入，这时候图片路径就相对要提上去2级
	Mat im = imread("../../00284.jpg");
	mtcnn find(im.cols, im.rows);
	vector<Rect> objs = find.detectObject(im);
	for (int i = 0; i < objs.size(); ++i)
		rectangle(im, objs[i], Scalar(0, 255), 2);

	imshow("demo", im);
	waitKey();
    return 0;
}