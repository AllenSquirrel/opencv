#include<iostream>
#include<cv.h>
#include<cxcore.h>
#include<highgui.h>
#include<windows.h>
using namespace std;

int main()
{
	int i = 0;
	int mode = CV_RETR_EXTERNAL;//设置为提取最外层轮廓的模式
	int counters_num = 0;//图像中提取轮廓的数目，默认设为0
	CvMemStorage *storage = cvCreateMemStorage(0);//提取轮廓是需要的存储容器，默认大小为0
	CvSeq *contour = 0;//存储提取轮廓的序列指针
	IplImage *src = cvLoadImage("F:\\visual c++&&opencv\\source\\timg.jpg");
	if (src != 0)
	{
		cvThreshold(src, src, 128, 255, CV_THRESH_BINARY);//二值化

		cvNamedWindow("src", 1);
		cvShowImage("bin_src", src);//显示二值化后图像

		counters_num = cvFindContours(src, storage, &contour, sizeof(CvContour), mode, CV_CHAIN_APPROX_NONE);
		cout << "检测出的轮廓数目" << counters_num << endl;

		IplImage* p = cvCreateImage(cvGetSize(src), IPL_DEPTH_8U, 3);
		CvSeqReader reader;
		int count = 0;
		if (contour != 0)
		{
			count = contour->total;
			cout << count << endl;
		}
		cvStartReadSeq(contour, &reader, 0);
		CvPoint pt1;
		CvScalar color = CV_RGB(255, 255, 255);
		cvNamedWindow("contour", 1);
		cvShowImage("contour", p);
		for (i = 0; i < count; i++)//逐点画轮廓
		{
			CV_READ_SEQ_ELEM(pt1, reader);//读取轮廓点
			cvCircle(p, pt1, 1, color);//根据点画轮廓
			cvShowImage("contour", p);
			cvWaitKey(5);
		}
		cvWaitKey(0);
		cvReleaseImage(&src);
		cvReleaseImage(&p);
		cvReleaseMemStorage(&storage);
	}
	system("pause");
	return 0;
}