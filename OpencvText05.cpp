#include<iostream>
#include<cv.h>
#include<highgui.h>
#include<windows.h>
using namespace std;

int main()
{
	IplImage *src = 0;
	IplImage *dst = 0;
	IplImage *src_c1;
	IplImage *src_c2;
	IplImage *src_c3;
	IplImage *repeat;
	src = cvLoadImage("F:\\visual c++&&opencv\\source\\timg.jpg");
	if (src != 0)
	{
		repeat = cvCreateImage(CvSize(src->width*2,(int)src->height*1.5),IPL_DEPTH_8U,src->nChannels);
		cvRepeat(src,repeat);//用原数组管道式填充输出数组
		cvNamedWindow("src", CV_WINDOW_AUTOSIZE);
		cvNamedWindow("dst", CV_WINDOW_AUTOSIZE);
		cvNamedWindow("repeat", CV_WINDOW_AUTOSIZE);
		cvShowImage("src", src);
		cvShowImage("repeat", repeat);

		cvWaitKey(1000);
		cvFlip(src,NULL,-1);
		cvShowImage("src_flip", src);

		src_c1 = cvCreateImage(CvSize(src->width, src->height), IPL_DEPTH_8U, 1);
		src_c2 = cvCreateImage(CvSize(src->width, src->height), IPL_DEPTH_8U, 1);
		src_c3 = cvCreateImage(CvSize(src->width, src->height), IPL_DEPTH_8U, 1);

		dst= cvCreateImage(CvSize(src->width, src->height), IPL_DEPTH_8U, 3);
		cvSplit(src,src_c1,src_c2,src_c3,NULL);//源彩色图分离为三个通道
		cvShowImage("src_c1", src_c1);
		cvShowImage("src_c2", src_c1);//显示该通道下源图像单通道灰度图
		cvShowImage("src_c3", src_c1);
		cvWaitKey(1000);

		cvMerge(src_c1, src_c2, src_c3, NULL,dst);
		cvShowImage("dst", dst);
		cvWaitKey(0);

		cvReleaseImage(&src);
		cvReleaseImage(&dst);
		cvReleaseImage(&src_c1);
		cvReleaseImage(&src_c2);
		cvReleaseImage(&src_c3);

		cvDestroyWindow("src");
		cvDestroyWindow("dst");
		cvDestroyWindow("repeat");
	}
	system("pause");
	return 0;
}