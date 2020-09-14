#include <iostream>
#include <opencv2\opencv.hpp>
#include<opencv.hpp>
using namespace std;
using namespace cv;
int main() {
	Mat threshold_output;//����һ�����ؾ������������ͼ�����
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	Mat src_gray;
	Mat frame;//
	//�Զ�����̬ѧԪ�ؽṹ
	cv::Mat element5(9, 9, CV_8U, cv::Scalar(1));//5*5�����Σ�8λuchar�ͣ�ȫ1�ṹԪ��  
	Scalar color = Scalar(0, 255, 0);
	cv::Mat closed;
	Rect rect;
	frame = imread("F:\\visual c++&&opencv\\source\\timg.jpg");
	imshow("src", frame);
	Mat image = frame.clone();
	cvWaitKey(1000);
	cvtColor(frame, src_gray, COLOR_BGR2GRAY);
	//ʹ��Canny����Ե  
	Canny(src_gray, threshold_output, 80, 126, (3, 3));
	//�߼���̬ѧ�����㺯��  
	cv::morphologyEx(threshold_output, closed, cv::MORPH_CLOSE, element5);
	imshow("canny", threshold_output);
	imshow("erode", closed);
	// Ѱ������������  
	findContours(closed, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_NONE, Point());
	cout << contours.size() << endl;
	for (int i = 0; i < contours.size(); i++)
	{

		for (int j = 0; j < contours[i].size(); j++)
		{

			cout << contours[i][j] << endl;
			//����������ÿ�����ص㲢�Ҹ�ֵ��ɫ
			//frame.at<Vec3b>(contours[i][j].y, contours[i][j].x) = (0, 0, 255);
		}

	}
	//ת�������㵽�������ο�
	rect = boundingRect(contours[0]);
	rectangle(image, rect, color,2);
	//�Ӿ��ο���ͼƬ
	imshow("rect", image);
	cvWaitKey(0);
	return 0;
}