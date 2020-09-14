//ªÒ»°ÕºœÒ Ù–‘«“ÕºœÒµπ÷√
#include<iostream>
#include<cv.h>
#include<highgui.h>
#include<windows.h>
using namespace std;

int main()
{
	IplImage *src = 0;
	IplImage *dst = 0;
	src = cvLoadImage("F:\\visual c++&&opencv\\source\\timg.jpg");
	if (src != 0)
	{
		cout << "*************************** "<< endl;
		cout << "image infomations" << endl;
		cout << "*************************** " << endl;
		cout << "image id:" << src->ID << endl;
		cout << "image size:" << src->nSize << endl;
		cout << "image nchannel:" << src->nChannels << endl;
		cout << "image depth:" << src->depth << endl;
		cout << "image width:" << src->width << endl;
		cout << "image height:" << src->height << endl;
		for (int i = 0; i < 200; i = i + 3)
		{
			cout << (int)(uchar)src->imageData[i] << " " << (int)(uchar)src->imageData[i + 1] << " " << (int)(uchar)src->imageData[i + 2] << endl;
		}
		dst = cvCreateImage(CvSize(src->width, src->height), src->depth, src->nChannels);
		cvConvertImage(src, dst, CV_CVTIMG_FLIP);
		cvNamedWindow("src", CV_WINDOW_AUTOSIZE);
		cvNamedWindow("dst", CV_WINDOW_AUTOSIZE);
		cvShowImage("src", src);
		cvShowImage("dst", dst);
		cvWaitKey(0);
		cvReleaseImage(&src);
		cvReleaseImage(&dst);
		cvDestroyWindow("src");
		cvDestroyWindow("dst");
	}
	system("pause");
}