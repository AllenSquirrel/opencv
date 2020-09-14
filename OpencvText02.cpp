//Í¼Ïñ±ßÔµ¼ì²â
#include<iostream>
#include<cv.h>
#include<cxcore.h>
#include<highgui.h>
#include<windows.h>
using namespace cv;

int main()
{
	IplImage* pImg = NULL;
	IplImage* pCannyImg = NULL;
	pImg = cvLoadImage("F:\\visual c++&&opencv\\source\\timg.jpg");
	if (pImg != 0)
	{
		pCannyImg = cvCreateImage(cvGetSize(pImg), IPL_DEPTH_8U, 1);
		cvCanny(pImg, pCannyImg, 50.150, 3);
		cvNamedWindow("CANNY", 1);
		cvShowImage("CANNY", pCannyImg);
		cvWaitKey(0);
		cvReleaseImage(&pCannyImg);
		cvDestroyWindow("CANNY");
	}
	system("pause");
	return 0;
}

