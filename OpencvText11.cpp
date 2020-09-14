#include <iostream>
#include <opencv2\opencv.hpp>
#include<opencv.hpp>
using namespace std;
using namespace cv;
int main() {
	Mat threshold_output;//创建一个像素矩阵类对象，用于图像输出
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	Mat src_gray;
	Mat frame;//
	//自定义形态学元素结构
	cv::Mat element5(9, 9, CV_8U, cv::Scalar(1));//5*5正方形，8位uchar型，全1结构元素  
	Scalar color = Scalar(0, 255, 0);
	cv::Mat closed;
	Rect rect;
	frame = imread("F:\\visual c++&&opencv\\source\\timg.jpg");
	imshow("src", frame);
	Mat image = frame.clone();
	cvWaitKey(1000);
	cvtColor(frame, src_gray, COLOR_BGR2GRAY);
	//使用Canny检测边缘  
	Canny(src_gray, threshold_output, 80, 126, (3, 3));
	//高级形态学闭运算函数  
	cv::morphologyEx(threshold_output, closed, cv::MORPH_CLOSE, element5);
	imshow("canny", threshold_output);
	imshow("erode", closed);
	// 寻找外轮廓轮廓  
	findContours(closed, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_NONE, Point());
	cout << contours.size() << endl;
	for (int i = 0; i < contours.size(); i++)
	{

		for (int j = 0; j < contours[i].size(); j++)
		{

			cout << contours[i][j] << endl;
			//遍历轮廓中每个像素点并且赋值颜色
			//frame.at<Vec3b>(contours[i][j].y, contours[i][j].x) = (0, 0, 255);
		}

	}
	//转换轮廓点到最大外矩形框
	rect = boundingRect(contours[0]);
	rectangle(image, rect, color,2);
	//加矩形框后的图片
	imshow("rect", image);
	cvWaitKey(0);
	return 0;
}
