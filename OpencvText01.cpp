//Í¼ÏñËõ·Å
#include<iostream>
#include<cv.h>
#include<highgui.h>
#include<windows.h>
using namespace cv;

int main()
{
	IplImage *src=0;
	IplImage *dst=0;
	double scale = 0.618;
	CvSize dst_size;
	src = cvLoadImage("F:\\visual c++&&opencv\\source\\timg.jpg");
	if (src != 0)
	{
		dst_size.height = (int)(src->height*scale);
		dst_size.width = (int)(src->width*scale);
		dst = cvCreateImage(dst_size,src->depth,src->nChannels);
		cvResize(src, dst, CV_INTER_LINEAR);
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
	return 0;
}