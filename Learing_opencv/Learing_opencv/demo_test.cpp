#include<iostream>
#include<opencv2/opencv.hpp>
#include<opencv2/highgui.hpp>
#include<demo.h>
//#include<windows.h>
using namespace std;
using namespace cv;


int main(int argc,char *argv)
{
	Mat src = imread("F:/������Ӿ�/visual c++&&opencv/source/timg.png");
	if (src.empty())
	{
		printf("image load error!");
		return -1;
	}
	namedWindow("����ͼƬ", WINDOW_AUTOSIZE);
	imshow("����ͼƬ", src);

	imagedemo op;
	op.colorspace_demo(src);
	waitKey(0);
	//system("pauses");
	return 0;
}

