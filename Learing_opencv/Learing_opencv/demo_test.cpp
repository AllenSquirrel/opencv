#include<iostream>
#include<opencv2/opencv.hpp>
#include<opencv2/highgui.hpp>
#include<demo.h>
//#include<windows.h>
using namespace std;
using namespace cv;


int main(int argc,char *argv)
{
	Mat src = imread("F:/计算机视觉/visual c++&&opencv/source/timg.png");
	if (src.empty())
	{
		printf("image load error!");
		return -1;
	}
	namedWindow("输入图片", WINDOW_AUTOSIZE);
	imshow("输入图片", src);

	imagedemo op;
	op.colorspace_demo(src);
	waitKey(0);
	//system("pauses");
	return 0;
}

