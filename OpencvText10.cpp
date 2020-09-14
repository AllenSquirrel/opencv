#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <time.h>
#include <string.h>
#include <windows.h>


using namespace cv;
using namespace std;


//******************************************************************

#define threshold_diff1 25 //设置简单帧差法阈值
#define threshold_diff2 25 //设置简单帧差法阈值

#define fx_l 654.53369  //左眼标定焦距
#define fy_l 656.99896

#define cx_l 667.29262  //左眼图像物理中心点坐标
#define cy_l 385.50491

#define cx_r 671.90655  //右眼图像物理中心点坐标
#define cy_r 450.14218

#define fx_r 787.37707  //右眼标定焦距
#define fy_r 783.00536

#define R {0.01297,0.02971,0.00340}  //标定旋转矩阵
#define T {-404.97875;-87.07289;272.21939}  //标定平移矩阵


//******************************************************************





int time_stamp()//时间戳函数,获取当前系统时间
{
	time_t rawtime;
	struct tm timeinfo;
	time(&rawtime);

	localtime_s(&timeinfo,&rawtime);
	
	int Year = timeinfo.tm_year + 1900;
	int Mon = timeinfo.tm_mon + 1;
	int Day = timeinfo.tm_mday;
	int Hour = timeinfo.tm_hour;
	int Min = timeinfo.tm_min;
	int Second = timeinfo.tm_sec;
	cout << "=== == == == == == == == == == == == == == == == == == == == ==" << endl;
	cout << Year << ":" << Mon << ":" << Day << "-" << Hour << ":" << Min << ":" << Second << endl;
	return 0;
}






vector<Point> match_feature1()    //轮廓形状特征匹配
{
	Mat img_gray;
	Mat mid_filer2;
	
	Mat img = imread("F:\\visual c++&&opencv\\source\\1.png", CV_LOAD_IMAGE_UNCHANGED);

	Mat draw_img = img.clone();
	//namedWindow("jpg", WINDOW_AUTOSIZE);
	//imshow("jpg",img);

	cvtColor(img, img_gray, CV_BGR2GRAY);
	//Sobel(img_gray, img_gray, CV_8U, 1, 0, 3, 0.4, 128);
	//threshold(img_gray, img_gray,1, 255, CV_THRESH_BINARY);//确保黑中找白
	adaptiveThreshold(img_gray, img_gray,255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 25, 5);//自适应阈值分割


	dilate(img_gray, img_gray, Mat()); erode(img_gray, img_gray, Mat());//膨胀和腐蚀处理，有效消除高亮噪声
	//medianBlur(img_gray, mid_filer2, 3);//中值滤波
	bitwise_not(img_gray, img_gray);
	
	/*namedWindow("jpg", WINDOW_AUTOSIZE);
	imshow("jpg", img_gray);*/

	vector<vector<Point>> contours1;
	findContours(img_gray, contours1, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
	drawContours(draw_img, contours1, -1, Scalar(0, 255, 0), 2, 8);

	namedWindow("目标模板1", WINDOW_NORMAL);
	imshow("目标模板1", draw_img);
	return contours1[0];
}


vector<Point> match_feature2()    //轮廓形状特征匹配
{
	Mat img_gray;
	Mat mid_filer2;

	Mat img = imread("F:\\visual c++&&opencv\\source\\2.png", CV_LOAD_IMAGE_UNCHANGED);

	Mat draw_img = img.clone();
	//namedWindow("jpg", WINDOW_AUTOSIZE);
	//imshow("jpg",img);

	cvtColor(img, img_gray, CV_BGR2GRAY);
	//Sobel(img_gray, img_gray, CV_8U, 1, 0, 3, 0.4, 128);
	//threshold(img_gray, img_gray,1, 255, CV_THRESH_BINARY);//确保黑中找白
	adaptiveThreshold(img_gray, img_gray, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 25, 5);//自适应阈值分割


	dilate(img_gray, img_gray, Mat()); erode(img_gray, img_gray, Mat());//膨胀和腐蚀处理，有效消除高亮噪声
	//medianBlur(img_gray, mid_filer2, 3);//中值滤波
	bitwise_not(img_gray, img_gray);

	/*namedWindow("jpg", WINDOW_AUTOSIZE);
	imshow("jpg", img_gray);*/

	vector<vector<Point>> contours1;
	findContours(img_gray, contours1, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
	drawContours(draw_img, contours1, -1, Scalar(0, 255, 0), 2, 8);

	namedWindow("目标模板2", WINDOW_NORMAL);
	imshow("目标模板2", draw_img);
	return contours1[0];
}







int main(int argc, unsigned char* argv[])
{
	//match_feature();

	double x_l, y_l;//图像坐标系下目标二维平面坐标
	double X, Y, Z;//左眼相机坐标系下目标三维空间坐标
	double f_l, f_r;//左眼焦距，右眼焦距

	f_l = (fx_l + fy_l) / 2;
	f_r = (fx_r + fy_r) / 2;


	Mat frame;
	Mat img_src1, img_src2, img_src3;//3帧法需要3帧图片
	Mat img_dst, gray1, gray2, gray3;
	Mat gray_diff1, gray_diff2;//存储2次相减的图片
	Mat gray_diff11, gray_diff12;
	Mat gray_diff21, gray_diff22;
	Mat gray;//用来显示前景的
	Mat mid_filer;   //中值滤波法后的照片
	bool pause = false;


	VideoCapture vido_file("F:\\visual c++&&opencv\\source\\008.mp4");//在这里改相应的文件名
	namedWindow("foreground", WINDOW_NORMAL);

	//---------------------------------------------------------------------
	//获取视频的宽度、高度、帧率、总的帧数
	int frameH = vido_file.get(CV_CAP_PROP_FRAME_HEIGHT); //获取帧高
	int frameW = vido_file.get(CV_CAP_PROP_FRAME_WIDTH);  //获取帧宽
	int fps = vido_file.get(CV_CAP_PROP_FPS);          //获取帧率
	int numFrames = vido_file.get(CV_CAP_PROP_FRAME_COUNT);  //获取整个帧数
	int num = numFrames;
	printf("video's \nwidth = %d\t height = %d\n video's FPS = %d\t nums = %d\n", frameW, frameH, fps, numFrames);
	//---------------------------------------------------------------------

	while(1)
	{
		

		vido_file >>frame;
		Mat matRotation = getRotationMatrix2D(Point(frame.cols / 2, frame.rows / 2), 270, 1);//获取图像中心点旋转矩阵
		// Rotate the image
		Mat matRotatedFrame;
		warpAffine(frame, matRotatedFrame, matRotation, frame.size());
		//apture >> frame;
		//imshow("src", matRotatedFrame);
		if (!false)
		{
			vido_file >> img_src1;
			if (&img_src1 == nullptr)
			{
				printf("获取帧失败");
				break;
			}
			cvtColor(img_src1, gray1, CV_BGR2GRAY);

			waitKey(3);//考虑到pc机处理速度，每隔33ms获取一帧图像，并将其转化为灰度图像分别处理
			vido_file >> img_src2;
			if (&img_src2 == nullptr) 
			{
				printf("获取帧失败");
				break;
			}
			cvtColor(img_src2, gray2, CV_BGR2GRAY);

			waitKey(3);
			vido_file >> img_src3;
			if (&img_src3 == nullptr) //需要判断视频结束时，获取帧失败的情况
			{
				printf("处理结束");
				break;
			}
			cvtColor(img_src3, gray3, CV_BGR2GRAY);


			Sobel(gray1, gray1, CV_8U, 1, 0, 3, 0.4, 128);//sobel算子计算混合图像差分，由于sobel算子结合了Gaussian平滑和微分，所以，其结果或多或少对噪声有一定鲁棒性
			Sobel(gray2, gray2, CV_8U, 1, 0, 3, 0.4, 128);
			Sobel(gray3, gray3, CV_8U, 1, 0, 3, 0.4, 128);



			subtract(gray2, gray1, gray_diff11);//第二帧减第一帧
			subtract(gray1, gray2, gray_diff12);
			add(gray_diff11, gray_diff12, gray_diff1);
			subtract(gray3, gray2, gray_diff21);//第三帧减第二帧
			subtract(gray2, gray3, gray_diff22);
			add(gray_diff21, gray_diff22, gray_diff2);

			for (int i = 0; i < gray_diff1.rows; i++)
				for (int j = 0; j < gray_diff1.cols; j++)
				{
					if (abs(gray_diff1.at<unsigned char>(i, j)) >= threshold_diff1)//这里模板参数一定要用unsigned char，否则就一直报错
						gray_diff1.at<unsigned char>(i, j) = 255;            //第一次相减阈值处理
					else gray_diff1.at<unsigned char>(i, j) = 0;

					if (abs(gray_diff2.at<unsigned char>(i, j)) >= threshold_diff2)//第二次相减阈值处理
						gray_diff2.at<unsigned char>(i, j) = 255;
					else gray_diff2.at<unsigned char>(i, j) = 0;
				}
			bitwise_and(gray_diff1, gray_diff2, gray);//三帧差法第三步，差分图像进行与运算

			dilate(gray, gray, Mat()); erode(gray, gray, Mat());//膨胀和腐蚀处理，有效消除高亮噪声

			medianBlur(gray, mid_filer, 3);//中值滤波
		
			//GaussianBlur(gray, mid_filer, Size(3, 3), 0, 0);


			Mat matRotation1 = getRotationMatrix2D(Point(mid_filer.cols / 2, mid_filer.rows / 2), 270, 1);
			// Rotate the image
			Mat matRotatedFrame1;
			warpAffine(mid_filer, matRotatedFrame1, matRotation1, mid_filer.size());
			//apture >> frame;
			imshow("foreground", matRotatedFrame1);

			//int cnts,nums;
			//(matRotatedFrame1,cnts,CV_RETR_EXTERNAL);
			vector<vector<Point>> contours;
			vector<Vec4i> hierarchy;
			findContours(matRotatedFrame1, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, Point());//寻找轮廓
			
			Rect bt_point;//rect类存贮所创建矩形对象的左上角和右下角坐标（图像坐标系下）
			Point p1, p2;
	        Mat imageContours = Mat::zeros(matRotatedFrame1.size(), CV_8UC1);
			Mat Contours = Mat::zeros(matRotatedFrame1.size(), CV_8UC1);  //绘制  


			line(matRotatedFrame, Point(frameW/2-160, frameH/2), Point(frameW / 2 + 160, frameH/2), Scalar(0, 0, 255), 3, 8);
			line(matRotatedFrame, Point(frameW / 2, frameH / 2-160), Point(frameW / 2, frameH / 2 + 160), Scalar(255, 0, 0), 3, 8);

			for (int i = 0; i < contours.size(); i++)
			{
				//contours[i]代表的是第i个轮廓，contours[i].size()代表的是第i个轮廓上所有的像素点数,根据轮廓像素点个数

				double a0 = matchShapes(match_feature1(), contours[i], CV_CONTOURS_MATCH_I1, 0)-3;
				double a1 = matchShapes(match_feature2(), contours[i], CV_CONTOURS_MATCH_I1, 0)-3;

				if ((a0<1 && contours[i].size() > 80 && contours[i].size() < 320) || (a1 < 1 && contours[i].size() > 80 && contours[i].size()))
				{
					drawContours(matRotatedFrame, contours, i, Scalar(0, 255, 0), 2, 8);
					bt_point = boundingRect(contours[i]);
					p1.x = bt_point.x;
					p1.y = bt_point.y;
					p2.x = bt_point.x + bt_point.width;
					p2.y = bt_point.y + bt_point.height;
					rectangle(matRotatedFrame, p1, p2, Scalar(0, 255, 0), 2);//矩形框出ROI区域


					//获取ROI矩形框中心坐标    ((p2.x-p1.x)/2,(p2.y-p1.y)/2),   用于后续目标坐标计算

					x_l = (p2.x - p1.x) / 2 + p1.x;
					y_l = (p2.y - p1.y) / 2 + p1.y;
					if (x_l >= frameW / 2)
					{
						if (y_l <= frameH / 2)//第一象限
						{
							x_l = (x_l - frameW / 2);
							y_l = (frameH / 2 - y_l);
						}
						else //第四象限
						{
							x_l = (x_l - frameW / 2);
							y_l = -(y_l - frameH /2);
						}
					}
					else
					{
						if (y_l <= frameH / 2)//第二象限
						{
							x_l = -(frameW / 2 - x_l);
							y_l = (frameH / 2 - y_l);
						}
						else //第三象限
						{
							x_l = -(frameW / 2 - x_l);
							y_l = -(y_l - frameH / 2);
						}
					}

					//空间坐标获取



					//获取时间戳，打印坐标信息
					time_stamp();
					cout << "左眼图像坐标系下目标坐标：" << "x_l:" << x_l << "y_l:" << y_l << endl;

				}

			}
		}
		namedWindow("跟踪识别监视器", WINDOW_AUTOSIZE);
		imshow("跟踪识别监视器", matRotatedFrame);
		if (cvWaitKey(33) >= 0)
			break;
		
	}
	return 0;
}
