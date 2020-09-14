#include<iostream>
#include<cv.h>
#include<cxcore.h>
#include<highgui.h>
#include<windows.h>
using namespace std;

int main()
{
	int i = 0;
	int mode = CV_RETR_EXTERNAL;//����Ϊ��ȡ�����������ģʽ
	int counters_num = 0;//ͼ������ȡ��������Ŀ��Ĭ����Ϊ0
	CvMemStorage *storage = cvCreateMemStorage(0);//��ȡ��������Ҫ�Ĵ洢������Ĭ�ϴ�СΪ0
	CvSeq *contour = 0;//�洢��ȡ����������ָ��
	IplImage *src = cvLoadImage("F:\\visual c++&&opencv\\source\\timg.jpg");
	if (src != 0)
	{
		cvThreshold(src, src, 128, 255, CV_THRESH_BINARY);//��ֵ��

		cvNamedWindow("src", 1);
		cvShowImage("bin_src", src);//��ʾ��ֵ����ͼ��

		counters_num = cvFindContours(src, storage, &contour, sizeof(CvContour), mode, CV_CHAIN_APPROX_NONE);
		cout << "������������Ŀ" << counters_num << endl;

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
		for (i = 0; i < count; i++)//��㻭����
		{
			CV_READ_SEQ_ELEM(pt1, reader);//��ȡ������
			cvCircle(p, pt1, 1, color);//���ݵ㻭����
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