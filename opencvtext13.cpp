#include <iostream>  
#include <opencv2/core/core.hpp>  
#include <opencv2/highgui/highgui.hpp>  

using namespace cv;

int main()
{
	// 读入一张图片（poyanghu缩小图）    
	Mat img = imread("F:\\计算机视觉\\visual c++&&opencv\\source\\1.png");
	// 创建一个名为 "图片"窗口    
	namedWindow("图片");
	// 在窗口中显示图片   
	imshow("图片", img);
	// 等待6000 ms后窗口自动关闭    
	waitKey(6000);
	return 0;
}