#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgproc/imgproc_c.h>
#include <time.h>
#include <string.h>
#include <windows.h>

using namespace cv;
using namespace std;


void drawCircle(Mat &input, const vector<Vec3f> &circles) {
	for (int i = 0; i < circles.size(); i++) {
		Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
		int radius = cvRound(circles[i][2]);
		circle(input, center, radius, Scalar(0, 255, 0), 2, 8, 0);
	}
}


int time_stamp()//时间戳函数,获取当前系统时间
{
	time_t rawtime;
	struct tm timeinfo;
	time(&rawtime);

	localtime_s(&timeinfo, &rawtime);

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


Mat calibration_l(Mat image)
{
	Size image_size = image.size();
	float intrinsic[3][3] = { 680.85971,			 0,		664.11201,
									  0,	681.10996,		399.06642,
									  0,			 0,				1 };
	float distortion[1][5] = { -0.16422, 0.00177, -0.00045, 0.00054, 0 };
	Mat intrinsic_matrix = Mat(3, 3, CV_32FC1, intrinsic);
	Mat distortion_coeffs = Mat(1, 5, CV_32FC1, distortion);
	Mat R = Mat::eye(3, 3, CV_32F);
	Mat mapx = Mat(image_size, CV_32FC1);
	Mat mapy = Mat(image_size, CV_32FC1);
	initUndistortRectifyMap(intrinsic_matrix, distortion_coeffs, R, intrinsic_matrix, image_size, CV_32FC1, mapx, mapy);
	Mat t = image.clone();
	cv::remap(image, t, mapx, mapy, INTER_LINEAR);
	return t;
}

Mat calibration_r(Mat image)
{
	Size image_size = image.size();
	float intrinsic[3][3] = { 677.28468,			 0,		673.29250,
										0,	677.72044,		408.16192,
										0,			 0,				1 };
	float distortion[1][5] = { -0.16044, 0.00379, -0.00034, -0.00083, 0 };
	Mat intrinsic_matrix = Mat(3, 3, CV_32FC1, intrinsic);
	Mat distortion_coeffs = Mat(1, 5, CV_32FC1, distortion);
	Mat R = Mat::eye(3, 3, CV_32F);
	Mat mapx = Mat(image_size, CV_32FC1);
	Mat mapy = Mat(image_size, CV_32FC1);
	initUndistortRectifyMap(intrinsic_matrix, distortion_coeffs, R, intrinsic_matrix, image_size, CV_32FC1, mapx, mapy);
	Mat t = image.clone();
	cv::remap(image, t, mapx, mapy, INTER_LINEAR);
	return t;
}

//左相机内参数矩阵
float leftIntrinsic[3][3] = { 680.85971,			 0,		664.11201,
									  0,	681.10996,		399.06642,
									  0,			 0,				1 };
//左相机畸变系数
float leftDistortion[1][5] = { -0.16422, 0.00177, -0.00045, 0.00054, 0 };
//左相机旋转矩阵
float leftRotation[3][3] = { 1,		        0,		        0,
								   0,		        1,		        0,
								   0,		        0,		        1 };
//左相机平移向量
float leftTranslation[1][3] = { 0,0,0 };

//右相机内参数矩阵
float rightIntrinsic[3][3] = { 677.28468,			 0,		673.29250,
										0,	677.72044,		408.16192,
										0,			 0,				1 };
//右相机畸变系数
float rightDistortion[1][5] = { -0.16044, 0.00379, -0.00034, -0.00083, 0 };
//右相机旋转矩阵
float rightRotation[3][3] = { 1,		        0.0005441,		        -0.0040,
							 -0.00057997,		        1,		        -0.0090,
							 0.0040,		        0.0090,		              1 };
//右相机平移向量
float rightTranslation[1][3] = { -119.8714, 0.2209, -0.0451 };




Point3f uv2xyz(Point2f uvLeft, Point2f uvRight)
{
	//  [u1]      |X|					  [u2]      |X|
	//Z*|v1| = Ml*|Y|					Z*|v2| = Mr*|Y|
	//  [ 1]      |Z|					  [ 1]      |Z|
	//			  |1|								|1|
	Mat mLeftRotation = Mat(3, 3, CV_32F, leftRotation);
	Mat mLeftTranslation = Mat(3, 1, CV_32F, leftTranslation);
	Mat mLeftRT = Mat(3, 4, CV_32F);//左相机M矩阵
	hconcat(mLeftRotation, mLeftTranslation, mLeftRT);
	Mat mLeftIntrinsic = Mat(3, 3, CV_32F, leftIntrinsic);
	Mat mLeftM = mLeftIntrinsic * mLeftRT;
	//cout<<"左相机M矩阵 = "<<endl<<mLeftM<<endl;

	Mat mRightRotation = Mat(3, 3, CV_32F, rightRotation);
	Mat mRightTranslation = Mat(3, 1, CV_32F, rightTranslation);
	Mat mRightRT = Mat(3, 4, CV_32F);//右相机M矩阵
	hconcat(mRightRotation, mRightTranslation, mRightRT);
	Mat mRightIntrinsic = Mat(3, 3, CV_32F, rightIntrinsic);
	Mat mRightM = mRightIntrinsic * mRightRT;
	//cout<<"右相机M矩阵 = "<<endl<<mRightM<<endl;

	//最小二乘法A矩阵
	Mat A = Mat(4, 3, CV_32F);
	A.at<float>(0, 0) = uvLeft.x * mLeftM.at<float>(2, 0) - mLeftM.at<float>(0, 0);
	A.at<float>(0, 1) = uvLeft.x * mLeftM.at<float>(2, 1) - mLeftM.at<float>(0, 1);
	A.at<float>(0, 2) = uvLeft.x * mLeftM.at<float>(2, 2) - mLeftM.at<float>(0, 2);

	A.at<float>(1, 0) = uvLeft.y * mLeftM.at<float>(2, 0) - mLeftM.at<float>(1, 0);
	A.at<float>(1, 1) = uvLeft.y * mLeftM.at<float>(2, 1) - mLeftM.at<float>(1, 1);
	A.at<float>(1, 2) = uvLeft.y * mLeftM.at<float>(2, 2) - mLeftM.at<float>(1, 2);

	A.at<float>(2, 0) = uvRight.x * mRightM.at<float>(2, 0) - mRightM.at<float>(0, 0);
	A.at<float>(2, 1) = uvRight.x * mRightM.at<float>(2, 1) - mRightM.at<float>(0, 1);
	A.at<float>(2, 2) = uvRight.x * mRightM.at<float>(2, 2) - mRightM.at<float>(0, 2);

	A.at<float>(3, 0) = uvRight.y * mRightM.at<float>(2, 0) - mRightM.at<float>(1, 0);
	A.at<float>(3, 1) = uvRight.y * mRightM.at<float>(2, 1) - mRightM.at<float>(1, 1);
	A.at<float>(3, 2) = uvRight.y * mRightM.at<float>(2, 2) - mRightM.at<float>(1, 2);

	//最小二乘法B矩阵
	Mat B = Mat(4, 1, CV_32F);
	B.at<float>(0, 0) = mLeftM.at<float>(0, 3) - uvLeft.x * mLeftM.at<float>(2, 3);
	B.at<float>(1, 0) = mLeftM.at<float>(1, 3) - uvLeft.y * mLeftM.at<float>(2, 3);
	B.at<float>(2, 0) = mRightM.at<float>(0, 3) - uvRight.x * mRightM.at<float>(2, 3);
	B.at<float>(3, 0) = mRightM.at<float>(1, 3) - uvRight.x * mRightM.at<float>(2, 3);

	Mat XYZ = Mat(3, 1, CV_32F);
	//采用SVD最小二乘法求解XYZ
	solve(A, B, XYZ, DECOMP_SVD);

	//cout<<"空间坐标为 = "<<endl<<XYZ<<endl;

	//世界坐标系中坐标
	Point3f world;
	world.x = XYZ.at<float>(0, 0);
	world.y = XYZ.at<float>(1, 0);
	world.z = XYZ.at<float>(2, 0);

	return world;
}


//************************************************************************************************************************************

int main(int argc, unsigned char* argv[])
{

	//match_feature();

	double x_l=0, y_l=0;//图像坐标系下目标二维平面坐标,(左上角为坐标原点)
	double x_r=0, y_r=0;

	double x_l_, y_l_;//图像坐标系下目标二维平面坐标,(中心点为坐标原点)
	double x_r_, y_r_;//图像坐标系下目标二维平面坐标,(中心点为坐标原点)


	double X, Y, Z;//左眼相机坐标系下目标三维空间坐标




	VideoCapture vido_file_l("F:\\visual c++&&opencv\\source\\calibration\\1.avi");//在这里改相应的文件名
	VideoCapture vido_file_r("F:\\visual c++&&opencv\\source\\calibration\\2.avi");//在这里改相应的文件名

	/*namedWindow("foreground_l", WINDOW_AUTOSIZE);
	namedWindow("foreground_r", WINDOW_AUTOSIZE);*/


	//---------------------------------------------------------------------
	//获取视频的宽度、高度、帧率、总的帧数
	int frameH_l = vido_file_l.get(CV_CAP_PROP_FRAME_HEIGHT); //获取帧高
	int frameW_l = vido_file_l.get(CV_CAP_PROP_FRAME_WIDTH);  //获取帧宽
	int fps_l = vido_file_l.get(CV_CAP_PROP_FPS);          //获取帧率
	int numFrames_l = vido_file_l.get(CV_CAP_PROP_FRAME_COUNT);  //获取整个帧数
	int num_l = numFrames_l;
	cout << "Left:" << endl;
	printf("video's \nwidth = %d\t height = %d\n video's FPS = %d\t nums = %d\n", frameW_l, frameH_l, fps_l, numFrames_l);
	//---------------------------------------------------------------------

	int frameH_r = vido_file_r.get(CV_CAP_PROP_FRAME_HEIGHT); //获取帧高
	int frameW_r = vido_file_r.get(CV_CAP_PROP_FRAME_WIDTH);  //获取帧宽
	int fps_r = vido_file_r.get(CV_CAP_PROP_FPS);          //获取帧率
	int numFrames_r = vido_file_r.get(CV_CAP_PROP_FRAME_COUNT);  //获取整个帧数
	int num_r = numFrames_r;
	cout << "Right:" << endl;
	printf("video's \nwidth = %d\t height = %d\n video's FPS = %d\t nums = %d\n", frameW_r, frameH_r, fps_r, numFrames_r);

	while(1)
	{ 
		Mat frame_l, frame_r;
		Mat img_src_l, img_src_r;
		Mat img_l, img_r;
		vido_file_l >> frame_l;
		vido_file_r >> frame_r;

		vector<Vec3f> circles1;//识别出来的圆，每一行是一个圆，第一列是圆心的x坐标，第二列是圆心的y坐标，第三列是圆的半径
		vector<Vec3f> circles2;
		//Mat result = imread("D:/code/map3.png");//结果图

		//蓝色的HSV值
		int low_H = 100, low_S = 43, low_V = 46;
		int High_H = 124, High_S = 255, High_V = 255;

		if (!false)
		{
			/*img_src_l = frame_l.clone();
			img_src_r = frame_r.clone();*/
			vido_file_l >> img_l;
			vido_file_r >> img_r;
			img_l=calibration_l(img_l);
			img_r = calibration_r(img_r);
			if (&img_l == nullptr || &img_r == nullptr)
			{
				cout << "获取失败" << endl;
				break;
			}
			cvtColor(img_l, img_src_l, COLOR_BGR2HSV);//从BGR->HSV
			cvtColor(img_r, img_src_r, COLOR_BGR2HSV);//从BGR->HSV

			inRange(img_src_l, Scalar(low_H, low_S, low_V), Scalar(High_H, High_S, High_V), img_src_l);//二值化
			inRange(img_src_r, Scalar(low_H, low_S, low_V), Scalar(High_H, High_S, High_V), img_src_r);//二值化

			//Reduce the noise so we avoid false circle detection
			GaussianBlur(img_src_l, img_src_l, Size(5, 3), 2, 2);
			GaussianBlur(img_src_r, img_src_r, Size(5, 3), 2, 2);

			line(frame_l, Point(frameW_l / 2 - 60, frameH_l / 2), Point(frameW_l / 2 + 60, frameH_l / 2), Scalar(0, 0, 255), 2, 8);
		    line(frame_l, Point(frameW_l / 2, frameH_l / 2 - 60), Point(frameW_l / 2, frameH_l / 2 + 60), Scalar(255, 0, 0), 2, 8);

			line(frame_r, Point(frameW_r / 2 - 60, frameH_r / 2), Point(frameW_r / 2 + 60, frameH_r / 2), Scalar(0, 0, 255), 2, 8);
			line(frame_r, Point(frameW_r / 2, frameH_r / 2 - 60), Point(frameW_r / 2, frameH_r / 2 + 60), Scalar(255, 0, 0), 2, 8);
			HoughCircles(img_src_l, circles1, CV_HOUGH_GRADIENT, 1, 30, 30, 20, 10, 60);//找圆,最后两个参数是圆半径范围，20是最小圆半径，30是最大圆半径
			HoughCircles(img_src_r, circles2, CV_HOUGH_GRADIENT, 1, 30, 30, 20, 10, 60);//找圆,最后两个参数是圆半径范围，20是最小圆半径，30是最大圆半径(1,30,30,20,10,60 )

			drawCircle(frame_l, circles1);//画圆
			drawCircle(frame_r, circles2);//画圆

			for (int i = 0; i < circles1.size(); i++)
			{
				x_l = 0, y_l = 0;
				x_l_ = 0, y_l_ = 0;
				x_l = circles1[i][0];
				y_l = circles1[i][1];
				if (x_l >= frameW_l / 2)
				{
					if (y_l <= frameH_l / 2)//第一象限
					{
						x_l_ = (x_l - frameW_l / 2);
						y_l_ = (frameH_l / 2 - y_l);
					}
					else //第四象限
					{
						x_l_ = (x_l - frameW_l / 2);
						y_l_ = -(y_l - frameH_l / 2);
					}
				}
				else
				{
					if (y_l <= frameH_l / 2)//第二象限
					{
						x_l_ = -(frameW_l / 2 - x_l);
						y_l_ = (frameH_l / 2 - y_l);
					}
					else //第三象限
					{
						x_l_ = -(frameW_l / 2 - x_l);
						y_l_ = -(y_l - frameH_l / 2);
					}
				}
			}
			for (int i = 0; i < circles2.size(); i++)
			{
				x_r = 0, y_r = 0;
				x_r_ = 0, y_r_ = 0;
				x_r = circles2[i][0];
				y_r = circles2[i][1];
				if (x_r >= frameW_r / 2)
				{
					if (y_r <= frameH_r / 2)//第一象限
					{
						x_r_ = (x_r - frameW_r / 2);
						y_r_ = (frameH_r / 2 - y_r);
					}
					else //第四象限
					{
						x_r_ = (x_r - frameW_r / 2);
						y_r_ = -(y_r - frameH_r / 2);
					}
				}
				else
				{
					if (y_r <= frameH_r / 2)//第二象限
					{
						x_r_ = -(frameW_r / 2 - x_r);
						y_r_ = (frameH_r / 2 - y_r);
					}
					else //第三象限
					{
						x_r_ = -(frameW_r / 2 - x_r);
						y_r_ = -(y_r - frameH_r / 2);
					}
				}
			}
			if (x_l != 0 && y_l != 0 && x_r != 0 && y_r != 0)
			{

			//引入双目视觉数学测量模型
			//计算目标点在左眼相机坐标系下的三维坐标（X,Y,Z）
				Point3f worldPoint;
				worldPoint = uv2xyz(Point2f(x_l, y_l), Point2f(x_r, y_r));

				X = worldPoint.x / 1000;
				Y =- worldPoint.y / 1000;
				Z = worldPoint.z / 1000;


				if (X < 2 && Y < 0.5 && Z < 2.5&& Z>0)
				{
					time_stamp();
					cout << "左眼图像坐标系下目标坐标：" << "x_l:" << x_l_ << " y_l:" << y_l_ << endl;
					cout << "右眼图像坐标系下目标坐标：" << "x_r:" << x_r_ << " y_r:" << y_r_ << endl;
					cout << "==*-*-*-*-*-*-*-*-*-*-*-*-*-*-*==" << endl;
					cout << "相机坐标系下三维坐标：" << "X:" << X << "m" << " Y:" << Y << "m" << " Z:" << Z << "m" << endl;
				}
			}


		}
		namedWindow("Display window1", WINDOW_NORMAL);//展示结果
		namedWindow("Display window2", WINDOW_NORMAL);
		resizeWindow("Display window1", 1240, 680);
		resizeWindow("Display window2", 1240, 680);
		imshow("Display window1", frame_l);
		imshow("Display window2", frame_r);
		if (cvWaitKey(33) >= 0) break;
	}


	return 0;
}


//while (1)
//{
//	bool flag = false;
//	/*Mat frame_l = imread("F:\\visual c++&&opencv\\source\\moudle\\1.png", CV_LOAD_IMAGE_UNCHANGED);
//	Mat frame_r = imread("F:\\visual c++&&opencv\\source\\moudle\\1.png", CV_LOAD_IMAGE_UNCHANGED);*/
//	Mat frame_l, frame_r;
//	Mat gray_l, gray_r;
//	Mat mid_filer_l, mid_filer_r;
//	Mat img_gray_l, img_gray_r;
//	Mat img_src_l, img_src_r;

//	Mat matHsv_l, matHsv_r;

//	vido_file_l >> frame_l;
//	vido_file_r >> frame_r;

//	if (!false)
//	{
//		/*img_src_l = frame_l.clone();
//		img_src_r = frame_r.clone();*/
//		vido_file_l >> img_src_l;
//		vido_file_r >> img_src_r;
//		if (&img_src_l == nullptr || &img_src_r == nullptr)
//		{
//			cout << "获取失败" << endl;
//			flag = true;
//			break;
//		}

//		cvtColor(img_src_l, gray_l, CV_BGR2GRAY);
//		cvtColor(img_src_r, gray_r, CV_BGR2GRAY);

//		cvtColor(img_src_l, matHsv_l, COLOR_BGR2HSV);
//		cvtColor(img_src_r, matHsv_r, COLOR_BGR2HSV);


//		vector<int> colorVec_l, colorVec_r;
//		colorVec_l.push_back(matHsv_l.at<uchar>(0, 0));
//		colorVec_l.push_back(matHsv_l.at<uchar>(0, 1));
//		colorVec_l.push_back(matHsv_l.at<uchar>(0, 2));

//		colorVec_r.push_back(matHsv_r.at<uchar>(0, 0));
//		colorVec_r.push_back(matHsv_r.at<uchar>(0, 1));
//		colorVec_r.push_back(matHsv_r.at<uchar>(0, 2));

//		//waitKey(33);//考虑到pc机处理速度，每隔33ms获取一帧图像，并将其转化为灰度图像分别处理

//		adaptiveThreshold(gray_l, img_gray_l, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 25, 5);//自适应阈值分割
//		adaptiveThreshold(gray_r, img_gray_r, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 25, 5);//自适应阈值分割

//		imshow("left", img_gray_l);
//		imshow("right", img_gray_r);
//		dilate(img_gray_l, gray_l, Mat()); erode(img_gray_l, gray_l, Mat());//膨胀和腐蚀处理，有效消除高亮噪声
//		dilate(img_gray_r, gray_r, Mat()); erode(img_gray_r, gray_r, Mat());

//		medianBlur(gray_l, mid_filer_l, 3);//中值滤波
//		medianBlur(gray_r, mid_filer_r, 3);//中值滤波

//		vector<vector<Point>> contours_l;
//		vector<Vec4i> hierarchy_l;
//		vector<vector<Point>> contours_r;
//		vector<Vec4i> hierarchy_r;

//		/*vector<vector<Point>> contours1_poly(contours_l.size());
//		vector<vector<Point>> contours2_poly(contours_r.size());*/

//		vector <Point> point_l;
//		vector <Point> point_r;

//		/*Point center_l;
//		Point center_r;*/

//		findContours(mid_filer_l, contours_l, hierarchy_l, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, Point());//寻找轮廓
//		findContours(mid_filer_r, contours_r, hierarchy_r, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, Point());//寻找轮廓

//		Rect bt_point_l, bt_point_r;//rect类存贮所创建矩形对象的左上角和右下角坐标（图像坐标系下）
//		Point p1_l, p2_l, p1_r, p2_r;

//		Mat imageContours_l = Mat::zeros(mid_filer_l.size(), CV_8UC1);
//		Mat imageContours_r = Mat::zeros(mid_filer_r.size(), CV_8UC1);

//		Mat Contours_l = Mat::zeros(mid_filer_l.size(), CV_8UC1);  //绘制  
//		Mat Contours_r = Mat::zeros(mid_filer_r.size(), CV_8UC1);  //绘制  


//		/*line(frame_l, Point(frameW_l / 2 - 60, frameH_l / 2), Point(frameW_l / 2 + 60, frameH_l / 2), Scalar(0, 0, 255), 2, 8);
//		line(frame_l, Point(frameW_l / 2, frameH_l / 2 - 60), Point(frameW_l / 2, frameH_l / 2 + 60), Scalar(255, 0, 0), 2, 8);

//		line(frame_r, Point(frameW_r / 2 - 60, frameH_r / 2), Point(frameW_r / 2 + 60, frameH_r / 2), Scalar(0, 0, 255), 2, 8);
//		line(frame_r, Point(frameW_r / 2, frameH_r / 2 - 60), Point(frameW_r / 2, frameH_r / 2 + 60), Scalar(255, 0, 0), 2, 8);*/


//		for (size_t i = 0; i < contours_l.size(); i++)
//		{

//			x_l = 0, y_l = 0;
//			x_l_ = 0, y_l_ = 0;
//			double epsilon1 = 0.01*arcLength(contours_l[i], true);

//			approxPolyDP(Mat(contours_l[i]), point_l, epsilon1, true);//轮廓逼近，确定角点个数判断目标形状

//			int corners1 = point_l.size();//圆

//			//HoughCircles(edges, circles, CV_HOUGH_GRADIENT, 1.5, 10, 200, 100, 0, 0);

//			//vector<Vec3f> circles_l;
//			//HoughCircles(mid_filer_l, circles_l, CV_HOUGH_GRADIENT, 1, mid_filer_l.rows / 20, 100, 100, 0, 0);
//			////依次在图中绘制出圆
//			//for (size_t i = 0; i < circles_l.size(); i++)
//			//{
//			//	Point center(cvRound(circles_l[i][0]), cvRound(circles_l[i][1]));
//			//	int radius = cvRound(circles_l[i][2]);
//			//	//绘制圆心
//			//	circle(frame_l, center, 3, Scalar(0, 255, 0), -1, 8, 0);
//			//	//绘制圆轮廓
//			//	circle(frame_l, center, radius, Scalar(155, 50, 255), 3, 8, 0);
//			//	flag = true;
//			//}


//			if ((colorVec_l[0] >= 100 && colorVec_l[0] <= 124)
//				&& (colorVec_l[1] >= 43 && colorVec_l[1] <= 255)
//				&& (colorVec_l[2] >= 46 && colorVec_l[2] <= 255) && corners1 > 15 && (contours_l[i].size() > 45 && contours_l[i].size() < 85))
//			{
//				//flag = false;
//				drawContours(frame_l, contours_l, i, Scalar(0, 255, 0), 2, 8);
//				bt_point_l = boundingRect(contours_l[i]);
//				p1_l.x = bt_point_l.x;
//				p1_l.y = bt_point_l.y;
//				p2_l.x = bt_point_l.x + bt_point_l.width;
//				p2_l.y = bt_point_l.y + bt_point_l.height;
//				rectangle(frame_l, p1_l, p2_l, Scalar(0, 255, 0), 2);//矩形框出ROI区域


//				//获取ROI矩形框中心坐标    ((p2.x-p1.x)/2,(p2.y-p1.y)/2),   用于后续目标坐标计算

//				x_l = (p2_l.x - p1_l.x) / 2 + p1_l.x;
//				y_l = (p2_l.y - p1_l.y) / 2 + p1_l.y;

//				/*x_l = center_l.x;
//				y_l = center_l.y;*/
//				if (x_l >= frameW_l / 2)
//				{
//					if (y_l <= frameH_l / 2)//第一象限
//					{
//						x_l_ = (x_l - frameW_l / 2);
//						y_l_ = (frameH_l / 2 - y_l);
//					}
//					else //第四象限
//					{
//						x_l_ = (x_l - frameW_l / 2);
//						y_l_ = -(y_l - frameH_l / 2);
//					}
//				}
//				else
//				{
//					if (y_l <= frameH_l / 2)//第二象限
//					{
//						x_l_ = -(frameW_l / 2 - x_l);
//						y_l_ = (frameH_l / 2 - y_l);
//					}
//					else //第三象限
//					{
//						x_l_ = -(frameW_l / 2 - x_l);
//						y_l_ = -(y_l - frameH_l / 2);
//					}
//				}

//			}

//		}
//		for (int i = 0; i < contours_r.size(); i++)
//		{
//			x_r = 0, y_r = 0;
//			x_r_ = 0, y_r_ = 0;
//			//contours[i]代表的是第i个轮廓，contours[i].size()代表的是第i个轮廓上所有的像素点数,根据轮廓像素点个数

//			double epsilon2 = 0.01*arcLength(contours_r[i], true);

//			approxPolyDP(contours_r[i], point_r, epsilon2, true);//轮廓逼近，确定角点个数判断目标形状

//			int corners2 = point_r.size();//圆
//			//vector<Vec3f> circles_r;
//			//HoughCircles(mid_filer_r, circles_r, CV_HOUGH_GRADIENT, 1, mid_filer_r.rows / 20, 100, 100, 0, 0);
//			////依次在图中绘制出圆
//			//for (size_t i = 0; i < circles_r.size(); i++)
//			//{
//			//	Point center(cvRound(circles_r[i][0]), cvRound(circles_r[i][1]));
//			//	int radius = cvRound(circles_r[i][2]);
//			//	//绘制圆心
//			//	circle(frame_r, center, 3, Scalar(0, 255, 0), -1, 8, 0);
//			//	//绘制圆轮廓
//			//	circle(frame_r, center, radius, Scalar(155, 50, 255), 3, 8, 0);
//			//	flag = true;
//			//}

//			if ((colorVec_r[0] >= 100 && colorVec_r[0] <= 124)
//				&& (colorVec_r[1] >= 43 && colorVec_r[1] <= 255)
//				&& (colorVec_r[2] >= 46 && colorVec_r[2] <= 255) && corners2 > 15 && (contours_r[i].size() > 45 && contours_r[i].size() < 85))
//			{
//				//flag = false;
//				//drawContours(frame, contours, i, Scalar(0, 255, 0), 2, 8);
//				drawContours(frame_r, contours_r, i, Scalar(0, 255, 0), 2, 8);
//				bt_point_r = boundingRect(contours_r[i]);
//				p1_r.x = bt_point_r.x;
//				p1_r.y = bt_point_r.y;
//				p2_r.x = bt_point_r.x + bt_point_r.width;
//				p2_r.y = bt_point_r.y + bt_point_r.height;
//				rectangle(frame_r, p1_r, p2_r, Scalar(0, 255, 0), 2);//矩形框出ROI区域


//				//获取ROI矩形框中心坐标    ((p2.x-p1.x)/2,(p2.y-p1.y)/2),   用于后续目标坐标计算

//				x_l = (p2_r.x - p1_r.x) / 2 + p1_r.x;
//				y_r = (p2_r.y - p1_r.y) / 2 + p1_r.y;

//				/*	x_l = center_r.x;
//					y_l = center_r.y;*/
//				if (x_r >= frameW_r / 2)
//				{
//					if (y_r <= frameH_r / 2)//第一象限
//					{
//						x_r_ = (x_r - frameW_r / 2);
//						y_r_ = (frameH_r / 2 - y_r);
//					}
//					else //第四象限
//					{
//						x_r_ = (x_r - frameW_r / 2);
//						y_r_ = -(y_r - frameH_r / 2);
//					}
//				}
//				else
//				{
//					if (y_r <= frameH_r / 2)//第二象限
//					{
//						x_r_ = -(frameW_r / 2 - x_r);
//						y_r_ = (frameH_r / 2 - y_r);
//					}
//					else //第三象限
//					{
//						x_r_ = -(frameW_r / 2 - x_r);
//						y_r_ = -(y_r - frameH_r / 2);
//					}
//				}

//				//空间坐标获取



//				//获取时间戳，打印坐标信息
//				if (x_l != 0 && y_l != 0 && x_r != 0 && y_r != 0)
//				{

//					//引入双目视觉数学测量模型
//					//计算目标点在左眼相机坐标系下的三维坐标（X,Y,Z）
//					Point3f worldPoint;
//					worldPoint = uv2xyz(Point2f(x_l, y_l), Point2f(x_r, y_r));

//					X = worldPoint.x / 1000;
//					Y = worldPoint.y / 1000;
//					Z = worldPoint.z / 1000;


//					time_stamp();
//					cout << "左眼图像坐标系下目标坐标：" << "x_l:" << x_l_ << " y_l:" << y_l_ << endl;
//					cout << "右眼图像坐标系下目标坐标：" << "x_r:" << x_r_ << " y_r:" << y_r_ << endl;
//					cout << "==*-*-*-*-*-*-*-*-*-*-*-*-*-*-*==" << endl;
//					cout << "相机坐标系下三维坐标：" << "X:" << X << "m" << " Y:" << Y << "m" << " Z:" << Z << "m" << endl;
//				}
//			}
//		}
//	}
//	namedWindow("左眼跟踪识别监视器", WINDOW_AUTOSIZE);
//	namedWindow("右眼跟踪识别监视器", WINDOW_AUTOSIZE);
//	imshow("左眼跟踪识别监视器", frame_l);
//	imshow("右眼跟踪识别监视器", frame_r);
//	if (cvWaitKey(33) >= 0) break;
//	if (flag) break;
//}
