#include<iostream>
#include<cv.h>
#include<highgui.h>
//ROI感兴趣区域获取
#include<windows.h>
using namespace std;

int main()
{
	IplImage *src = 0;
	IplImage *dst = 0;
	src = cvLoadImage("F:\\visual c++&&opencv\\source\\timg.jpg");
	CvRect ROI_src;
	CvRect ROI_dst;
	if (src != 0)
	{
		cvNamedWindow("src", CV_WINDOW_AUTOSIZE);
		cvMoveWindow("src", 200, 200);
		cvShowImage("src", src);//设置ROI前原始图像
		cvWaitKey(800);
		ROI_src.x = 250;
		ROI_src.y = 250;
		ROI_src.width=200;
		ROI_src.height = 200;
		cvSetImageROI(src,ROI_src);
		cout << "源图像ROI：" << endl;
		cout << "x=" << src->roi->xOffset << "  " << "y=" << src->roi->yOffset << endl;
		cout << "width=" << src->roi->width << "  " << "height=" << src->roi->height << endl;
		cvShowImage("src_roi", src);//设置ROI后原始图像

		dst = cvCloneImage(src);
		//cvShowImage("dst", dst);
		ROI_dst = cvGetImageROI(dst);
		cout << "目标图像ROI：" << endl;
		cout << "x=" << dst->roi->xOffset << "  " << "y=" << dst->roi->yOffset << endl;
		cout << "width=" << dst->roi->width << "  " << "height=" << dst->roi->height << endl;
		cvNamedWindow("dst", CV_WINDOW_AUTOSIZE);
		cvMoveWindow("dst", 400, 200);
		cvShowImage("dst_roi", dst);
		cvWaitKey(800);
		cvResetImageROI(dst);
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