
#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <time.h>
#include <string.h>
#include <windows.h>
#include <GL/glut.h>

#include <thread>
#include <future>

#pragma comment(linker,"/subsystem:\"windows\" /entry:\"mainCRTStartup\"")



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

#define R {0.01297;0.02971;0.00340}  //标定旋转矩阵
#define T {-404.97875;-87.07289;272.21939}  //标定平移矩阵

#define r1 0.01297
#define r4 0.02971
#define r7 0.00340

#define tx -404.97875
#define ty -87.07289
#define tz 272.21939
//******************************************************************
#define MAX_CHAR        128

void drawString(const char* str) {
	static int isFirstCall = 1;
	static GLuint lists;

	if (isFirstCall) { // 如果是第一次调用，执行初始化
						 // 为每一个ASCII字符产生一个显示列表
		isFirstCall = 0;

		// 申请MAX_CHAR个连续的显示列表编号
		lists = glGenLists(MAX_CHAR);

		// 把每个字符的绘制命令都装到对应的显示列表中
		wglUseFontBitmaps(wglGetCurrentDC(), 0, MAX_CHAR, lists);
	}
	// 调用每个字符对应的显示列表，绘制每个字符
	for (; *str != '\0'; ++str)
		glCallList(lists + *str);
}


//*******
//用来画一个坐标轴的函数
void axis(double length)
{
	glColor3f(1.0f, 0.0f, 0.0f);
	glPushMatrix();
	glLineWidth(4.f);//宽度
	glBegin(GL_LINES);
	glVertex3d(0.0, 0.0, 0.0);
	glVertex3d(0.0, 0.0, length);//先画一轴，Z轴
	glEnd();
	//将当前操作点移到指定位置
	glTranslated(0.0, 0.0, length - 0.01);
	glColor3f(1.0, 0.0, 0.0);
	glutSolidCone(0.02, -0.1, 8, 8);
	glPopMatrix();
}
void paint(double x, double y, double z)
{
	glClearColor(1, 1, 1, 0);
	glClear(GL_COLOR_BUFFER_BIT);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();

	glOrtho(-0.5, 2.0, -2.0, 2.0, -100, 100);//调整坐标轴位置
	glPointSize(4);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	gluLookAt(1.3, 1.6, 2.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);

	//画坐标系
	axis(-2);

	glPushMatrix();
	glRotated(-90.0, 0, 1.0, 0);//绕y轴-方向旋转90度，即x轴
	axis(-2);
	glPopMatrix();


	glPushMatrix();
	glRotated(-90.0, 1.0, 0.0, 0.0);//绕x轴-方向旋转，y轴
	axis(-2);
	glPopMatrix();

	
	glColor3f(0.0f, 0.0f, 0.0f);
	glRasterPos3d(0, 0, -2.18);
	drawString("Z");
	glRasterPos3d(0, -2.18, 0);
	drawString("Y");
	glRasterPos3d(2.18, 0, 0);
	drawString("X");
	glRasterPos3d(0, -0.12, 0);
	drawString("O");

	GLfloat pointSize = 20.0f;
	glPointSize(pointSize);
	/*glColor3f(0.0f, 0.0f, 1.0f);*/
	glBegin(GL_POINTS);
	glColor3f(0.0f, 0.0f, 1.0f);
	glVertex3d(x, -y, -z);
	glEnd();

	glutSwapBuffers();

}
void Init()
{
	glClearColor(1.0, 1.0, 1.0, 1.0);
}

//*************





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






vector<Point> match_feature1()    //轮廓形状特征匹配1
{
	int max_contours = 1;
	int num;
	Mat img_gray;
	Mat mid_filer2;

	Mat img = imread("F:\\visual c++&&opencv\\source\\moudle\\1.png", CV_LOAD_IMAGE_UNCHANGED);

	Mat draw_img = img.clone();
	//namedWindow("jpg", WINDOW_AUTOSIZE);
	//imshow("jpg",img);

	cvtColor(img, img_gray, CV_BGR2GRAY);
	//Sobel(img_gray, img_gray, CV_8U, 1, 0, 3, 0.4, 128);
	//threshold(img_gray, img_gray,1, 255, CV_THRESH_BINARY);//确保黑中找白
	adaptiveThreshold(img_gray, img_gray, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 25, 5);//自适应阈值分割

	medianBlur(img_gray, mid_filer2, 3);//中值滤波
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

	//             max_conours
	for (int i = 0; i < size(contours1); i++)
	{
		if (size(contours1[i]) > max_contours)
		{
			max_contours = size(contours1[i]);
			num = i;
		}
	}
	return contours1[num];
}


vector<Point> match_feature2()    //轮廓形状特征匹配2
{
	int max_contours = 1;
	int num;
	Mat img_gray;
	Mat mid_filer2;

	Mat img = imread("F:\\visual c++&&opencv\\source\\moudle\\2.png", CV_LOAD_IMAGE_UNCHANGED);

	Mat draw_img = img.clone();
	//namedWindow("jpg", WINDOW_AUTOSIZE);
	//imshow("jpg",img);

	cvtColor(img, img_gray, CV_BGR2GRAY);
	//Sobel(img_gray, img_gray, CV_8U, 1, 0, 3, 0.4, 128);
	//threshold(img_gray, img_gray,1, 255, CV_THRESH_BINARY);//确保黑中找白
	adaptiveThreshold(img_gray, img_gray, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 25, 5);//自适应阈值分割

	medianBlur(img_gray, mid_filer2, 3);//中值滤波
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

	//             max_conours
	for (int i = 0; i < size(contours1); i++)
	{
		if (size(contours1[i]) > max_contours)
		{
			max_contours = size(contours1[i]);
			num = i;
		}
	}
	return contours1[num];
}


vector<Point> match_feature3()    //轮廓形状特征匹配3
{
	int max_contours = 1;
	int num;
	Mat img_gray;
	Mat mid_filer2;

	Mat img = imread("F:\\visual c++&&opencv\\source\\moudle\\3.png", CV_LOAD_IMAGE_UNCHANGED);

	Mat draw_img = img.clone();
	//namedWindow("jpg", WINDOW_AUTOSIZE);
	//imshow("jpg",img);

	cvtColor(img, img_gray, CV_BGR2GRAY);
	//Sobel(img_gray, img_gray, CV_8U, 1, 0, 3, 0.4, 128);
	//threshold(img_gray, img_gray,1, 255, CV_THRESH_BINARY);//确保黑中找白
	adaptiveThreshold(img_gray, img_gray, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 25, 5);//自适应阈值分割

	medianBlur(img_gray, mid_filer2, 3);//中值滤波
	dilate(img_gray, img_gray, Mat()); erode(img_gray, img_gray, Mat());//膨胀和腐蚀处理，有效消除高亮噪声
	//medianBlur(img_gray, mid_filer2, 3);//中值滤波
	bitwise_not(img_gray, img_gray);

	/*namedWindow("jpg", WINDOW_AUTOSIZE);
	imshow("jpg", img_gray);*/

	vector<vector<Point>> contours1;
	findContours(img_gray, contours1, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
	drawContours(draw_img, contours1, -1, Scalar(0, 255, 0), 2, 8);

	namedWindow("目标模板3", WINDOW_NORMAL);
	imshow("目标模板3", draw_img);

	//             max_conours
	for (int i = 0; i < size(contours1); i++)
	{
		if (size(contours1[i]) > max_contours)
		{
			max_contours = size(contours1[i]);
			num = i;
		}
	}
	return contours1[num];
}


vector<Point> match_feature4()    //轮廓形状特征匹配4
{
	int max_contours = 1;
	int num;
	Mat img_gray;
	Mat mid_filer2;

	Mat img = imread("F:\\visual c++&&opencv\\source\\moudle\\4.png", CV_LOAD_IMAGE_UNCHANGED);

	Mat draw_img = img.clone();
	//namedWindow("jpg", WINDOW_AUTOSIZE);
	//imshow("jpg",img);

	cvtColor(img, img_gray, CV_BGR2GRAY);
	//Sobel(img_gray, img_gray, CV_8U, 1, 0, 3, 0.4, 128);
	//threshold(img_gray, img_gray,1, 255, CV_THRESH_BINARY);//确保黑中找白
	adaptiveThreshold(img_gray, img_gray, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 25, 5);//自适应阈值分割

	medianBlur(img_gray, mid_filer2, 3);//中值滤波
	dilate(img_gray, img_gray, Mat()); erode(img_gray, img_gray, Mat());//膨胀和腐蚀处理，有效消除高亮噪声
	//medianBlur(img_gray, mid_filer2, 3);//中值滤波
	bitwise_not(img_gray, img_gray);

	/*namedWindow("jpg", WINDOW_AUTOSIZE);
	imshow("jpg", img_gray);*/

	vector<vector<Point>> contours1;
	findContours(img_gray, contours1, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
	drawContours(draw_img, contours1, -1, Scalar(0, 255, 0), 2, 8);

	namedWindow("目标模板4", WINDOW_NORMAL);
	imshow("目标模板4", draw_img);

	//             max_conours
	for (int i = 0; i < size(contours1); i++)
	{
		if (size(contours1[i]) > max_contours)
		{
			max_contours = size(contours1[i]);
			num = i;
		}
	}
	return contours1[num];
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
	B.at<float>(3, 0) = mRightM.at<float>(1, 3) - uvRight.y * mRightM.at<float>(2, 3);

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

int main(int argc, char* argv[])
{

	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE);
	glutInitWindowSize(800, 800);
	glutInitWindowPosition(1150, 10);
	glutCreateWindow("3D空间目标运动动态显示");
	Init();

	/*glutMainLoop();*/

	//开4个线程，用于模板匹配匹配处理，节省处理时间

	std::packaged_task<vector<Point>()>mypt1(match_feature1);//函数mythread通过packaged_task包装
	std::thread feature_obj1(std::ref(mypt1));
	std::packaged_task<vector<Point>()>mypt2(match_feature2);
	std::thread feature_obj2(std::ref(mypt2));
	std::packaged_task<vector<Point>()>mypt3(match_feature3);
	std::thread feature_obj3(std::ref(mypt3));
	std::packaged_task<vector<Point>()>mypt4(match_feature4);
	std::thread feature_obj4(std::ref(mypt4));



	//match_feature();

	double x_l, y_l;//图像坐标系下目标二维平面坐标,(左上角为坐标原点)
	double x_r, y_r;

	double x_l_, y_l_;//图像坐标系下目标二维平面坐标,(中心点为坐标原点)
	double x_r_, y_r_;//图像坐标系下目标二维平面坐标,(中心点为坐标原点)


	double X, Y, Z;//左眼相机坐标系下目标三维空间坐标


	Mat frame_l;
	Mat frame_r;
	Mat img_src1_l, img_src2_l, img_src3_l;//左眼3帧法需要3帧图片
	Mat img_src1_r, img_src2_r, img_src3_r;//右眼3帧法需要3帧图片

	Mat img_dst_l, gray1_l, gray2_l, gray3_l;
	Mat img_dst_r, gray1_r, gray2_r, gray3_r;

	Mat gray_diff1_l, gray_diff2_l;//存储2次相减的图片
	Mat gray_diff1_r, gray_diff2_r;//存储2次相减的图片

	Mat gray_diff11_l, gray_diff12_l;
	Mat gray_diff11_r, gray_diff12_r;

	Mat gray_diff21_l, gray_diff22_l;
	Mat gray_diff21_r, gray_diff22_r;

	Mat gray_l, gray_r;//用来显示前景的
	Mat mid_filer_l;   //中值滤波法后的照片
	Mat mid_filer_r;   //中值滤波法后的照片
	bool pause = false;


	VideoCapture vido_file_l("F:\\visual c++&&opencv\\source\\text_l.avi");//在这里改相应的文件名
	VideoCapture vido_file_r("F:\\visual c++&&opencv\\source\\text_r.avi");//在这里改相应的文件名
	namedWindow("foreground_l", WINDOW_AUTOSIZE);
	namedWindow("foreground_r", WINDOW_AUTOSIZE);


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


	feature_obj1.join();
	feature_obj2.join();
	feature_obj3.join();
	feature_obj4.join();

	std::future<vector<Point>>result1 = mypt1.get_future();//std::future对象result通过借助packaged_task类的对象mypt来保存线程入口函数返回值（future类和packaged_task类绑定）
	std::future<vector<Point>>result2 = mypt2.get_future();//std::future对象result通过借助packaged_task类的对象mypt来保存线程入口函数返回值（future类和packaged_task类绑定）
	std::future<vector<Point>>result3 = mypt3.get_future();//std::future对象result通过借助packaged_task类的对象mypt来保存线程入口函数返回值（future类和packaged_task类绑定）
	std::future<vector<Point>>result4 = mypt4.get_future();//std::future对象result通过借助packaged_task类的对象mypt来保存线程入口函数返回值（future类和packaged_task类绑定）

	std::shared_future<vector<Point>>result_s1(std::move(result1));
	std::shared_future<vector<Point>>result_s2(std::move(result2));
	std::shared_future<vector<Point>>result_s3(std::move(result3));
	std::shared_future<vector<Point>>result_s4(std::move(result4));

	while (1)
	{


		vido_file_l >> frame_l;
		vido_file_r >> frame_r;
		//Mat matRotation = getRotationMatrix2D(Point(frame.cols / 2, frame.rows / 2), 270, 1);//获取图像中心点旋转矩阵
		//Mat matRotatedFrame;// Rotate the image
		//warpAffine(frame, matRotatedFrame, matRotation, frame.size());

		/*apture >> frame;*/
		//imshow("src", matRotatedFrame);
		if (!false)
		{
			vido_file_l >> img_src1_l;
			vido_file_r >> img_src1_r;
			if (&img_src1_l == nullptr || &img_src1_r == nullptr)
			{
				printf("获取帧失败");
				break;
			}
			cvtColor(img_src1_l, gray1_l, CV_BGR2GRAY);
			cvtColor(img_src1_r, gray1_r, CV_BGR2GRAY);

			waitKey(33);//考虑到pc机处理速度，每隔33ms获取一帧图像，并将其转化为灰度图像分别处理

			vido_file_l >> img_src2_l;
			vido_file_r >> img_src2_r;
			if (&img_src2_l == nullptr || &img_src2_r == nullptr)
			{
				printf("获取帧失败");
				break;
			}
			cvtColor(img_src2_l, gray2_l, CV_BGR2GRAY);
			cvtColor(img_src2_r, gray2_r, CV_BGR2GRAY);

			waitKey(33);

			vido_file_l >> img_src3_l;
			vido_file_r >> img_src3_r;
			if (&img_src3_l == nullptr || &img_src3_r == nullptr) //需要判断视频结束时，获取帧失败的情况
			{
				printf("处理结束");
				break;
			}
			cvtColor(img_src3_l, gray3_l, CV_BGR2GRAY);
			cvtColor(img_src3_r, gray3_r, CV_BGR2GRAY);

			Sobel(gray1_l, gray1_l, CV_8U, 1, 0, 3, 0.4, 128);//sobel算子计算混合图像差分，由于sobel算子结合了Gaussian平滑和微分，所以，其结果或多或少对噪声有一定鲁棒性
			Sobel(gray1_r, gray1_r, CV_8U, 1, 0, 3, 0.4, 128);

			Sobel(gray2_l, gray2_l, CV_8U, 1, 0, 3, 0.4, 128);
			Sobel(gray2_r, gray2_r, CV_8U, 1, 0, 3, 0.4, 128);

			Sobel(gray3_l, gray3_l, CV_8U, 1, 0, 3, 0.4, 128);
			Sobel(gray3_r, gray3_r, CV_8U, 1, 0, 3, 0.4, 128);


			subtract(gray2_l, gray1_l, gray_diff11_l);//第二帧减第一帧
			subtract(gray2_r, gray1_r, gray_diff11_r);

			subtract(gray1_l, gray2_l, gray_diff12_l);
			subtract(gray1_r, gray2_r, gray_diff12_r);

			add(gray_diff11_l, gray_diff12_l, gray_diff1_l);
			add(gray_diff11_r, gray_diff12_r, gray_diff1_r);

			subtract(gray3_l, gray2_l, gray_diff21_l);//第三帧减第二帧
			subtract(gray3_r, gray2_r, gray_diff21_r);

			subtract(gray2_l, gray3_l, gray_diff22_l);
			subtract(gray2_r, gray3_r, gray_diff22_r);

			add(gray_diff21_l, gray_diff22_l, gray_diff2_l);
			add(gray_diff21_r, gray_diff22_r, gray_diff2_r);


			for (int i = 0; i < gray_diff1_l.rows; i++)
				for (int j = 0; j < gray_diff1_l.cols; j++)
				{
					if (abs(gray_diff1_l.at<unsigned char>(i, j)) >= threshold_diff1)//这里模板参数一定要用unsigned char，否则就一直报错
						gray_diff1_l.at<unsigned char>(i, j) = 255;            //第一次相减阈值处理
					else gray_diff1_l.at<unsigned char>(i, j) = 0;

					if (abs(gray_diff2_l.at<unsigned char>(i, j)) >= threshold_diff2)//第二次相减阈值处理
						gray_diff2_l.at<unsigned char>(i, j) = 255;
					else gray_diff2_l.at<unsigned char>(i, j) = 0;
				}
			for (int i = 0; i < gray_diff1_r.rows; i++)
				for (int j = 0; j < gray_diff1_r.cols; j++)
				{
					if (abs(gray_diff1_r.at<unsigned char>(i, j)) >= threshold_diff1)//这里模板参数一定要用unsigned char，否则就一直报错
						gray_diff1_r.at<unsigned char>(i, j) = 255;            //第一次相减阈值处理
					else gray_diff1_r.at<unsigned char>(i, j) = 0;

					if (abs(gray_diff2_r.at<unsigned char>(i, j)) >= threshold_diff2)//第二次相减阈值处理
						gray_diff2_r.at<unsigned char>(i, j) = 255;
					else gray_diff2_r.at<unsigned char>(i, j) = 0;
				}

			bitwise_and(gray_diff1_l, gray_diff2_l, gray_l);//三帧差法第三步，差分图像进行与运算
			bitwise_and(gray_diff1_r, gray_diff2_r, gray_r);

			dilate(gray_l, gray_l, Mat()); erode(gray_l, gray_l, Mat());//膨胀和腐蚀处理，有效消除高亮噪声
			dilate(gray_r, gray_r, Mat()); erode(gray_r, gray_r, Mat());

			medianBlur(gray_l, mid_filer_l, 3);//中值滤波
			medianBlur(gray_r, mid_filer_r, 3);//中值滤波
			//GaussianBlur(gray, mid_filer, Size(3, 3), 0, 0);


			//Mat matRotation1 = getRotationMatrix2D(Point(mid_filer.cols / 2, mid_filer.rows / 2), 270, 1);
			//Mat matRotatedFrame1;// Rotate the image
			//warpAffine(mid_filer, matRotatedFrame1, matRotation1, mid_filer.size());
			//apture >> frame;
			imshow("foreground_l", mid_filer_l);
			imshow("foreground_r", mid_filer_r);


			//int cnts,nums;
			//(matRotatedFrame1,cnts,CV_RETR_EXTERNAL);
			vector<vector<Point>> contours_l;
			vector<Vec4i> hierarchy_l;
			vector<vector<Point>> contours_r;
			vector<Vec4i> hierarchy_r;

			vector <Point> point_l;
			vector <Point> point_r;

			findContours(mid_filer_l, contours_l, hierarchy_l, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, Point());//寻找轮廓
			findContours(mid_filer_r, contours_r, hierarchy_r, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, Point());//寻找轮廓

			Rect bt_point_l, bt_point_r;//rect类存贮所创建矩形对象的左上角和右下角坐标（图像坐标系下）
			Point p1_l, p2_l, p1_r, p2_r;

			Mat imageContours_l = Mat::zeros(mid_filer_l.size(), CV_8UC1);
			Mat imageContours_r = Mat::zeros(mid_filer_r.size(), CV_8UC1);

			Mat Contours_l = Mat::zeros(mid_filer_l.size(), CV_8UC1);  //绘制  
			Mat Contours_r = Mat::zeros(mid_filer_r.size(), CV_8UC1);  //绘制  


			line(frame_l, Point(frameW_l / 2 - 60, frameH_l / 2), Point(frameW_l / 2 + 60, frameH_l / 2), Scalar(0, 0, 255), 2, 8);
			line(frame_l, Point(frameW_l / 2, frameH_l / 2 - 60), Point(frameW_l / 2, frameH_l / 2 + 60), Scalar(255, 0, 0), 2, 8);

			line(frame_r, Point(frameW_r / 2 - 60, frameH_r / 2), Point(frameW_r / 2 + 60, frameH_r / 2), Scalar(0, 0, 255), 2, 8);
			line(frame_r, Point(frameW_r / 2, frameH_r / 2 - 60), Point(frameW_r / 2, frameH_r / 2 + 60), Scalar(255, 0, 0), 2, 8);


			for (int i = 0; i < contours_l.size(); i++)
			{

				x_l = 0, y_l = 0;
				x_l_ = 0, y_l_ = 0;
				//contours[i]代表的是第i个轮廓，contours[i].size()代表的是第i个轮廓上所有的像素点数,根据轮廓像素点个数

				double a0 = matchShapes(result_s1.get(), contours_l[i], CV_CONTOURS_MATCH_I3, 0) - 3;
				double a1 = matchShapes(result_s2.get(), contours_l[i], CV_CONTOURS_MATCH_I3, 0) - 3;
				double a2 = matchShapes(result_s3.get(), contours_l[i], CV_CONTOURS_MATCH_I3, 0) - 3;
				double a3 = matchShapes(result_s4.get(), contours_l[i], CV_CONTOURS_MATCH_I3, 0) - 3;

				double epsilon1 = 0.01*arcLength(contours_l[i], true);
				approxPolyDP(Mat(contours_l[i]), point_l, epsilon1, true);
				int corners1 = point_l.size();
				if ((a0 < 1 || a1 < 1 || a2 < 1 || a3 < 1) && corners1 >= 5 && (contours_l[i].size() > 20 && contours_l[i].size() < 300))
				{
					//drawContours(frame, contours, i, Scalar(0, 255, 0), 2, 8);
					bt_point_l = boundingRect(contours_l[i]);
					p1_l.x = bt_point_l.x;
					p1_l.y = bt_point_l.y;
					p2_l.x = bt_point_l.x + bt_point_l.width;
					p2_l.y = bt_point_l.y + bt_point_l.height;
					rectangle(frame_l, p1_l, p2_l, Scalar(0, 255, 0), 2);//矩形框出ROI区域


					//获取ROI矩形框中心坐标    ((p2.x-p1.x)/2,(p2.y-p1.y)/2),   用于后续目标坐标计算

					x_l = (p2_l.x - p1_l.x) / 2 + p1_l.x;
					y_l = (p2_l.y - p1_l.y) / 2 + p1_l.y;
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

					//空间坐标获取



					//获取时间戳，打印坐标信息
					/*time_stamp();
					cout << "左眼图像坐标系下目标坐标：" << "x_l:" << x_l << "y_l:" << y_l << endl;*/

				}

			}
			for (int i = 0; i < contours_r.size(); i++)
			{
				x_r = 0, y_r = 0;
				x_r_ = 0, y_r_ = 0;
				//contours[i]代表的是第i个轮廓，contours[i].size()代表的是第i个轮廓上所有的像素点数,根据轮廓像素点个数

				double a0 = matchShapes(result_s1.get(), contours_r[i], CV_CONTOURS_MATCH_I3, 0) - 3;
				double a1 = matchShapes(result_s2.get(), contours_r[i], CV_CONTOURS_MATCH_I3, 0) - 3;
				double a2 = matchShapes(result_s3.get(), contours_r[i], CV_CONTOURS_MATCH_I3, 0) - 3;
				double a3 = matchShapes(result_s4.get(), contours_r[i], CV_CONTOURS_MATCH_I3, 0) - 3;

				double epsilon2 = 0.01*arcLength(contours_r[i], true);
				approxPolyDP(Mat(contours_r[i]), point_r, epsilon2, true);
				int corners2 = point_r.size();
				if ((a0 < 1 || a1 < 1 || a2 < 1 || a3 < 1) && corners2 >= 5 && (contours_r[i].size() > 20 && contours_r[i].size() < 300))
				{
					//drawContours(frame, contours, i, Scalar(0, 255, 0), 2, 8);
					bt_point_r = boundingRect(contours_r[i]);
					p1_r.x = bt_point_r.x;
					p1_r.y = bt_point_r.y;
					p2_r.x = bt_point_r.x + bt_point_r.width;
					p2_r.y = bt_point_r.y + bt_point_r.height;
					rectangle(frame_r, p1_r, p2_r, Scalar(0, 255, 0), 2);//矩形框出ROI区域


					//获取ROI矩形框中心坐标    ((p2.x-p1.x)/2,(p2.y-p1.y)/2),   用于后续目标坐标计算

					x_r = (p2_r.x - p1_r.x) / 2 + p1_r.x;
					y_r = (p2_r.y - p1_r.y) / 2 + p1_r.y;
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

					//空间坐标获取



					//获取时间戳，打印坐标信息
					if (x_l != 0 && y_l != 0 && x_r != 0 && y_r != 0)
					{

						//引入双目视觉数学测量模型
						//计算目标点在左眼相机坐标系下的三维坐标（X,Y,Z）
						Point3f worldPoint;
						worldPoint = uv2xyz(Point2f(x_l, y_l), Point2f(x_r, y_r));

						X = worldPoint.x / 1000;
						Y = - worldPoint.y / 1000;
						Z = worldPoint.z / 1000;




						paint(X / 15, Y / 15, Z / 15);//量化后描点
						/*glutMainLoop();*/

						time_stamp();
						cout << "左眼图像坐标系下目标坐标：" << "x_l:" << x_l_ << " y_l:" << y_l_ << endl;
						cout << "右眼图像坐标系下目标坐标：" << "x_r:" << x_r_ << " y_r:" << y_r_ << endl;
						cout << "==*-*-*-*-*-*-*-*-*-*-*-*-*-*-*==" << endl;
						cout << "相机坐标系下三维坐标：" << "X:" << X << "m" << " Y:" << Y << "m" << " Z:" << Z << "m" << endl;
					}
				}

			}
		}
		namedWindow("左眼跟踪识别监视器", WINDOW_AUTOSIZE);
		imshow("左眼跟踪识别监视器", frame_l);
		namedWindow("右眼跟踪识别监视器", WINDOW_AUTOSIZE);
		imshow("右眼跟踪识别监视器", frame_r);

		if (cvWaitKey(33) >= 0)
			break;


	}
	glutMainLoop();
	return 0;
}








