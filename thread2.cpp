
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

#define threshold_diff1 25 //���ü�֡���ֵ
#define threshold_diff2 25 //���ü�֡���ֵ

#define fx_l 654.53369  //���۱궨����
#define fy_l 656.99896

#define cx_l 667.29262  //����ͼ���������ĵ�����
#define cy_l 385.50491

#define cx_r 671.90655  //����ͼ���������ĵ�����
#define cy_r 450.14218

#define fx_r 787.37707  //���۱궨����
#define fy_r 783.00536

#define R {0.01297;0.02971;0.00340}  //�궨��ת����
#define T {-404.97875;-87.07289;272.21939}  //�궨ƽ�ƾ���

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

	if (isFirstCall) { // ����ǵ�һ�ε��ã�ִ�г�ʼ��
						 // Ϊÿһ��ASCII�ַ�����һ����ʾ�б�
		isFirstCall = 0;

		// ����MAX_CHAR����������ʾ�б���
		lists = glGenLists(MAX_CHAR);

		// ��ÿ���ַ��Ļ������װ����Ӧ����ʾ�б���
		wglUseFontBitmaps(wglGetCurrentDC(), 0, MAX_CHAR, lists);
	}
	// ����ÿ���ַ���Ӧ����ʾ�б�����ÿ���ַ�
	for (; *str != '\0'; ++str)
		glCallList(lists + *str);
}


//*******
//������һ��������ĺ���
void axis(double length)
{
	glColor3f(1.0f, 0.0f, 0.0f);
	glPushMatrix();
	glLineWidth(4.f);//���
	glBegin(GL_LINES);
	glVertex3d(0.0, 0.0, 0.0);
	glVertex3d(0.0, 0.0, length);//�Ȼ�һ�ᣬZ��
	glEnd();
	//����ǰ�������Ƶ�ָ��λ��
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

	glOrtho(-0.5, 2.0, -2.0, 2.0, -100, 100);//����������λ��
	glPointSize(4);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	gluLookAt(1.3, 1.6, 2.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);

	//������ϵ
	axis(-2);

	glPushMatrix();
	glRotated(-90.0, 0, 1.0, 0);//��y��-������ת90�ȣ���x��
	axis(-2);
	glPopMatrix();


	glPushMatrix();
	glRotated(-90.0, 1.0, 0.0, 0.0);//��x��-������ת��y��
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





int time_stamp()//ʱ�������,��ȡ��ǰϵͳʱ��
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






vector<Point> match_feature1()    //������״����ƥ��1
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
	//threshold(img_gray, img_gray,1, 255, CV_THRESH_BINARY);//ȷ�������Ұ�
	adaptiveThreshold(img_gray, img_gray, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 25, 5);//����Ӧ��ֵ�ָ�

	medianBlur(img_gray, mid_filer2, 3);//��ֵ�˲�
	dilate(img_gray, img_gray, Mat()); erode(img_gray, img_gray, Mat());//���ͺ͸�ʴ������Ч������������
	//medianBlur(img_gray, mid_filer2, 3);//��ֵ�˲�
	bitwise_not(img_gray, img_gray);

	/*namedWindow("jpg", WINDOW_AUTOSIZE);
	imshow("jpg", img_gray);*/

	vector<vector<Point>> contours1;
	findContours(img_gray, contours1, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
	drawContours(draw_img, contours1, -1, Scalar(0, 255, 0), 2, 8);

	namedWindow("Ŀ��ģ��1", WINDOW_NORMAL);
	imshow("Ŀ��ģ��1", draw_img);

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


vector<Point> match_feature2()    //������״����ƥ��2
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
	//threshold(img_gray, img_gray,1, 255, CV_THRESH_BINARY);//ȷ�������Ұ�
	adaptiveThreshold(img_gray, img_gray, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 25, 5);//����Ӧ��ֵ�ָ�

	medianBlur(img_gray, mid_filer2, 3);//��ֵ�˲�
	dilate(img_gray, img_gray, Mat()); erode(img_gray, img_gray, Mat());//���ͺ͸�ʴ������Ч������������
	//medianBlur(img_gray, mid_filer2, 3);//��ֵ�˲�
	bitwise_not(img_gray, img_gray);

	/*namedWindow("jpg", WINDOW_AUTOSIZE);
	imshow("jpg", img_gray);*/

	vector<vector<Point>> contours1;
	findContours(img_gray, contours1, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
	drawContours(draw_img, contours1, -1, Scalar(0, 255, 0), 2, 8);

	namedWindow("Ŀ��ģ��2", WINDOW_NORMAL);
	imshow("Ŀ��ģ��2", draw_img);

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


vector<Point> match_feature3()    //������״����ƥ��3
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
	//threshold(img_gray, img_gray,1, 255, CV_THRESH_BINARY);//ȷ�������Ұ�
	adaptiveThreshold(img_gray, img_gray, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 25, 5);//����Ӧ��ֵ�ָ�

	medianBlur(img_gray, mid_filer2, 3);//��ֵ�˲�
	dilate(img_gray, img_gray, Mat()); erode(img_gray, img_gray, Mat());//���ͺ͸�ʴ������Ч������������
	//medianBlur(img_gray, mid_filer2, 3);//��ֵ�˲�
	bitwise_not(img_gray, img_gray);

	/*namedWindow("jpg", WINDOW_AUTOSIZE);
	imshow("jpg", img_gray);*/

	vector<vector<Point>> contours1;
	findContours(img_gray, contours1, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
	drawContours(draw_img, contours1, -1, Scalar(0, 255, 0), 2, 8);

	namedWindow("Ŀ��ģ��3", WINDOW_NORMAL);
	imshow("Ŀ��ģ��3", draw_img);

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


vector<Point> match_feature4()    //������״����ƥ��4
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
	//threshold(img_gray, img_gray,1, 255, CV_THRESH_BINARY);//ȷ�������Ұ�
	adaptiveThreshold(img_gray, img_gray, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 25, 5);//����Ӧ��ֵ�ָ�

	medianBlur(img_gray, mid_filer2, 3);//��ֵ�˲�
	dilate(img_gray, img_gray, Mat()); erode(img_gray, img_gray, Mat());//���ͺ͸�ʴ������Ч������������
	//medianBlur(img_gray, mid_filer2, 3);//��ֵ�˲�
	bitwise_not(img_gray, img_gray);

	/*namedWindow("jpg", WINDOW_AUTOSIZE);
	imshow("jpg", img_gray);*/

	vector<vector<Point>> contours1;
	findContours(img_gray, contours1, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
	drawContours(draw_img, contours1, -1, Scalar(0, 255, 0), 2, 8);

	namedWindow("Ŀ��ģ��4", WINDOW_NORMAL);
	imshow("Ŀ��ģ��4", draw_img);

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



//������ڲ�������
float leftIntrinsic[3][3] = { 680.85971,			 0,		664.11201,
									  0,	681.10996,		399.06642,
									  0,			 0,				1 };
//���������ϵ��
float leftDistortion[1][5] = { -0.16422, 0.00177, -0.00045, 0.00054, 0 };
//�������ת����
float leftRotation[3][3] = { 1,		        0,		        0,
								   0,		        1,		        0,
								   0,		        0,		        1 };
//�����ƽ������
float leftTranslation[1][3] = { 0,0,0 };

//������ڲ�������
float rightIntrinsic[3][3] = { 677.28468,			 0,		673.29250,
										0,	677.72044,		408.16192,
										0,			 0,				1 };
//���������ϵ��
float rightDistortion[1][5] = { -0.16044, 0.00379, -0.00034, -0.00083, 0 };
//�������ת����
float rightRotation[3][3] = { 1,		        0.0005441,		        -0.0040,
							 -0.00057997,		        1,		        -0.0090,
							 0.0040,		        0.0090,		              1 };
//�����ƽ������
float rightTranslation[1][3] = { -119.8714, 0.2209, -0.0451 };




Point3f uv2xyz(Point2f uvLeft, Point2f uvRight)
{
	//  [u1]      |X|					  [u2]      |X|
	//Z*|v1| = Ml*|Y|					Z*|v2| = Mr*|Y|
	//  [ 1]      |Z|					  [ 1]      |Z|
	//			  |1|								|1|
	Mat mLeftRotation = Mat(3, 3, CV_32F, leftRotation);
	Mat mLeftTranslation = Mat(3, 1, CV_32F, leftTranslation);
	Mat mLeftRT = Mat(3, 4, CV_32F);//�����M����
	hconcat(mLeftRotation, mLeftTranslation, mLeftRT);
	Mat mLeftIntrinsic = Mat(3, 3, CV_32F, leftIntrinsic);
	Mat mLeftM = mLeftIntrinsic * mLeftRT;
	//cout<<"�����M���� = "<<endl<<mLeftM<<endl;

	Mat mRightRotation = Mat(3, 3, CV_32F, rightRotation);
	Mat mRightTranslation = Mat(3, 1, CV_32F, rightTranslation);
	Mat mRightRT = Mat(3, 4, CV_32F);//�����M����
	hconcat(mRightRotation, mRightTranslation, mRightRT);
	Mat mRightIntrinsic = Mat(3, 3, CV_32F, rightIntrinsic);
	Mat mRightM = mRightIntrinsic * mRightRT;
	//cout<<"�����M���� = "<<endl<<mRightM<<endl;

	//��С���˷�A����
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

	//��С���˷�B����
	Mat B = Mat(4, 1, CV_32F);
	B.at<float>(0, 0) = mLeftM.at<float>(0, 3) - uvLeft.x * mLeftM.at<float>(2, 3);
	B.at<float>(1, 0) = mLeftM.at<float>(1, 3) - uvLeft.y * mLeftM.at<float>(2, 3);
	B.at<float>(2, 0) = mRightM.at<float>(0, 3) - uvRight.x * mRightM.at<float>(2, 3);
	B.at<float>(3, 0) = mRightM.at<float>(1, 3) - uvRight.y * mRightM.at<float>(2, 3);

	Mat XYZ = Mat(3, 1, CV_32F);
	//����SVD��С���˷����XYZ
	solve(A, B, XYZ, DECOMP_SVD);

	//cout<<"�ռ�����Ϊ = "<<endl<<XYZ<<endl;

	//��������ϵ������
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
	glutCreateWindow("3D�ռ�Ŀ���˶���̬��ʾ");
	Init();

	/*glutMainLoop();*/

	//��4���̣߳�����ģ��ƥ��ƥ�䴦����ʡ����ʱ��

	std::packaged_task<vector<Point>()>mypt1(match_feature1);//����mythreadͨ��packaged_task��װ
	std::thread feature_obj1(std::ref(mypt1));
	std::packaged_task<vector<Point>()>mypt2(match_feature2);
	std::thread feature_obj2(std::ref(mypt2));
	std::packaged_task<vector<Point>()>mypt3(match_feature3);
	std::thread feature_obj3(std::ref(mypt3));
	std::packaged_task<vector<Point>()>mypt4(match_feature4);
	std::thread feature_obj4(std::ref(mypt4));



	//match_feature();

	double x_l, y_l;//ͼ������ϵ��Ŀ���άƽ������,(���Ͻ�Ϊ����ԭ��)
	double x_r, y_r;

	double x_l_, y_l_;//ͼ������ϵ��Ŀ���άƽ������,(���ĵ�Ϊ����ԭ��)
	double x_r_, y_r_;//ͼ������ϵ��Ŀ���άƽ������,(���ĵ�Ϊ����ԭ��)


	double X, Y, Z;//�����������ϵ��Ŀ����ά�ռ�����


	Mat frame_l;
	Mat frame_r;
	Mat img_src1_l, img_src2_l, img_src3_l;//����3֡����Ҫ3֡ͼƬ
	Mat img_src1_r, img_src2_r, img_src3_r;//����3֡����Ҫ3֡ͼƬ

	Mat img_dst_l, gray1_l, gray2_l, gray3_l;
	Mat img_dst_r, gray1_r, gray2_r, gray3_r;

	Mat gray_diff1_l, gray_diff2_l;//�洢2�������ͼƬ
	Mat gray_diff1_r, gray_diff2_r;//�洢2�������ͼƬ

	Mat gray_diff11_l, gray_diff12_l;
	Mat gray_diff11_r, gray_diff12_r;

	Mat gray_diff21_l, gray_diff22_l;
	Mat gray_diff21_r, gray_diff22_r;

	Mat gray_l, gray_r;//������ʾǰ����
	Mat mid_filer_l;   //��ֵ�˲��������Ƭ
	Mat mid_filer_r;   //��ֵ�˲��������Ƭ
	bool pause = false;


	VideoCapture vido_file_l("F:\\visual c++&&opencv\\source\\text_l.avi");//���������Ӧ���ļ���
	VideoCapture vido_file_r("F:\\visual c++&&opencv\\source\\text_r.avi");//���������Ӧ���ļ���
	namedWindow("foreground_l", WINDOW_AUTOSIZE);
	namedWindow("foreground_r", WINDOW_AUTOSIZE);


	//---------------------------------------------------------------------
	//��ȡ��Ƶ�Ŀ�ȡ��߶ȡ�֡�ʡ��ܵ�֡��
	int frameH_l = vido_file_l.get(CV_CAP_PROP_FRAME_HEIGHT); //��ȡ֡��
	int frameW_l = vido_file_l.get(CV_CAP_PROP_FRAME_WIDTH);  //��ȡ֡��
	int fps_l = vido_file_l.get(CV_CAP_PROP_FPS);          //��ȡ֡��
	int numFrames_l = vido_file_l.get(CV_CAP_PROP_FRAME_COUNT);  //��ȡ����֡��
	int num_l = numFrames_l;
	cout << "Left:" << endl;
	printf("video's \nwidth = %d\t height = %d\n video's FPS = %d\t nums = %d\n", frameW_l, frameH_l, fps_l, numFrames_l);
	//---------------------------------------------------------------------

	int frameH_r = vido_file_r.get(CV_CAP_PROP_FRAME_HEIGHT); //��ȡ֡��
	int frameW_r = vido_file_r.get(CV_CAP_PROP_FRAME_WIDTH);  //��ȡ֡��
	int fps_r = vido_file_r.get(CV_CAP_PROP_FPS);          //��ȡ֡��
	int numFrames_r = vido_file_r.get(CV_CAP_PROP_FRAME_COUNT);  //��ȡ����֡��
	int num_r = numFrames_r;
	cout << "Right:" << endl;
	printf("video's \nwidth = %d\t height = %d\n video's FPS = %d\t nums = %d\n", frameW_r, frameH_r, fps_r, numFrames_r);


	feature_obj1.join();
	feature_obj2.join();
	feature_obj3.join();
	feature_obj4.join();

	std::future<vector<Point>>result1 = mypt1.get_future();//std::future����resultͨ������packaged_task��Ķ���mypt�������߳���ں�������ֵ��future���packaged_task��󶨣�
	std::future<vector<Point>>result2 = mypt2.get_future();//std::future����resultͨ������packaged_task��Ķ���mypt�������߳���ں�������ֵ��future���packaged_task��󶨣�
	std::future<vector<Point>>result3 = mypt3.get_future();//std::future����resultͨ������packaged_task��Ķ���mypt�������߳���ں�������ֵ��future���packaged_task��󶨣�
	std::future<vector<Point>>result4 = mypt4.get_future();//std::future����resultͨ������packaged_task��Ķ���mypt�������߳���ں�������ֵ��future���packaged_task��󶨣�

	std::shared_future<vector<Point>>result_s1(std::move(result1));
	std::shared_future<vector<Point>>result_s2(std::move(result2));
	std::shared_future<vector<Point>>result_s3(std::move(result3));
	std::shared_future<vector<Point>>result_s4(std::move(result4));

	while (1)
	{


		vido_file_l >> frame_l;
		vido_file_r >> frame_r;
		//Mat matRotation = getRotationMatrix2D(Point(frame.cols / 2, frame.rows / 2), 270, 1);//��ȡͼ�����ĵ���ת����
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
				printf("��ȡ֡ʧ��");
				break;
			}
			cvtColor(img_src1_l, gray1_l, CV_BGR2GRAY);
			cvtColor(img_src1_r, gray1_r, CV_BGR2GRAY);

			waitKey(33);//���ǵ�pc�������ٶȣ�ÿ��33ms��ȡһ֡ͼ�񣬲�����ת��Ϊ�Ҷ�ͼ��ֱ���

			vido_file_l >> img_src2_l;
			vido_file_r >> img_src2_r;
			if (&img_src2_l == nullptr || &img_src2_r == nullptr)
			{
				printf("��ȡ֡ʧ��");
				break;
			}
			cvtColor(img_src2_l, gray2_l, CV_BGR2GRAY);
			cvtColor(img_src2_r, gray2_r, CV_BGR2GRAY);

			waitKey(33);

			vido_file_l >> img_src3_l;
			vido_file_r >> img_src3_r;
			if (&img_src3_l == nullptr || &img_src3_r == nullptr) //��Ҫ�ж���Ƶ����ʱ����ȡ֡ʧ�ܵ����
			{
				printf("�������");
				break;
			}
			cvtColor(img_src3_l, gray3_l, CV_BGR2GRAY);
			cvtColor(img_src3_r, gray3_r, CV_BGR2GRAY);

			Sobel(gray1_l, gray1_l, CV_8U, 1, 0, 3, 0.4, 128);//sobel���Ӽ�����ͼ���֣�����sobel���ӽ����Gaussianƽ����΢�֣����ԣ����������ٶ�������һ��³����
			Sobel(gray1_r, gray1_r, CV_8U, 1, 0, 3, 0.4, 128);

			Sobel(gray2_l, gray2_l, CV_8U, 1, 0, 3, 0.4, 128);
			Sobel(gray2_r, gray2_r, CV_8U, 1, 0, 3, 0.4, 128);

			Sobel(gray3_l, gray3_l, CV_8U, 1, 0, 3, 0.4, 128);
			Sobel(gray3_r, gray3_r, CV_8U, 1, 0, 3, 0.4, 128);


			subtract(gray2_l, gray1_l, gray_diff11_l);//�ڶ�֡����һ֡
			subtract(gray2_r, gray1_r, gray_diff11_r);

			subtract(gray1_l, gray2_l, gray_diff12_l);
			subtract(gray1_r, gray2_r, gray_diff12_r);

			add(gray_diff11_l, gray_diff12_l, gray_diff1_l);
			add(gray_diff11_r, gray_diff12_r, gray_diff1_r);

			subtract(gray3_l, gray2_l, gray_diff21_l);//����֡���ڶ�֡
			subtract(gray3_r, gray2_r, gray_diff21_r);

			subtract(gray2_l, gray3_l, gray_diff22_l);
			subtract(gray2_r, gray3_r, gray_diff22_r);

			add(gray_diff21_l, gray_diff22_l, gray_diff2_l);
			add(gray_diff21_r, gray_diff22_r, gray_diff2_r);


			for (int i = 0; i < gray_diff1_l.rows; i++)
				for (int j = 0; j < gray_diff1_l.cols; j++)
				{
					if (abs(gray_diff1_l.at<unsigned char>(i, j)) >= threshold_diff1)//����ģ�����һ��Ҫ��unsigned char�������һֱ����
						gray_diff1_l.at<unsigned char>(i, j) = 255;            //��һ�������ֵ����
					else gray_diff1_l.at<unsigned char>(i, j) = 0;

					if (abs(gray_diff2_l.at<unsigned char>(i, j)) >= threshold_diff2)//�ڶ��������ֵ����
						gray_diff2_l.at<unsigned char>(i, j) = 255;
					else gray_diff2_l.at<unsigned char>(i, j) = 0;
				}
			for (int i = 0; i < gray_diff1_r.rows; i++)
				for (int j = 0; j < gray_diff1_r.cols; j++)
				{
					if (abs(gray_diff1_r.at<unsigned char>(i, j)) >= threshold_diff1)//����ģ�����һ��Ҫ��unsigned char�������һֱ����
						gray_diff1_r.at<unsigned char>(i, j) = 255;            //��һ�������ֵ����
					else gray_diff1_r.at<unsigned char>(i, j) = 0;

					if (abs(gray_diff2_r.at<unsigned char>(i, j)) >= threshold_diff2)//�ڶ��������ֵ����
						gray_diff2_r.at<unsigned char>(i, j) = 255;
					else gray_diff2_r.at<unsigned char>(i, j) = 0;
				}

			bitwise_and(gray_diff1_l, gray_diff2_l, gray_l);//��֡������������ͼ�����������
			bitwise_and(gray_diff1_r, gray_diff2_r, gray_r);

			dilate(gray_l, gray_l, Mat()); erode(gray_l, gray_l, Mat());//���ͺ͸�ʴ������Ч������������
			dilate(gray_r, gray_r, Mat()); erode(gray_r, gray_r, Mat());

			medianBlur(gray_l, mid_filer_l, 3);//��ֵ�˲�
			medianBlur(gray_r, mid_filer_r, 3);//��ֵ�˲�
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

			findContours(mid_filer_l, contours_l, hierarchy_l, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, Point());//Ѱ������
			findContours(mid_filer_r, contours_r, hierarchy_r, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, Point());//Ѱ������

			Rect bt_point_l, bt_point_r;//rect��������������ζ�������ϽǺ����½����꣨ͼ������ϵ�£�
			Point p1_l, p2_l, p1_r, p2_r;

			Mat imageContours_l = Mat::zeros(mid_filer_l.size(), CV_8UC1);
			Mat imageContours_r = Mat::zeros(mid_filer_r.size(), CV_8UC1);

			Mat Contours_l = Mat::zeros(mid_filer_l.size(), CV_8UC1);  //����  
			Mat Contours_r = Mat::zeros(mid_filer_r.size(), CV_8UC1);  //����  


			line(frame_l, Point(frameW_l / 2 - 60, frameH_l / 2), Point(frameW_l / 2 + 60, frameH_l / 2), Scalar(0, 0, 255), 2, 8);
			line(frame_l, Point(frameW_l / 2, frameH_l / 2 - 60), Point(frameW_l / 2, frameH_l / 2 + 60), Scalar(255, 0, 0), 2, 8);

			line(frame_r, Point(frameW_r / 2 - 60, frameH_r / 2), Point(frameW_r / 2 + 60, frameH_r / 2), Scalar(0, 0, 255), 2, 8);
			line(frame_r, Point(frameW_r / 2, frameH_r / 2 - 60), Point(frameW_r / 2, frameH_r / 2 + 60), Scalar(255, 0, 0), 2, 8);


			for (int i = 0; i < contours_l.size(); i++)
			{

				x_l = 0, y_l = 0;
				x_l_ = 0, y_l_ = 0;
				//contours[i]������ǵ�i��������contours[i].size()������ǵ�i�����������е����ص���,�����������ص����

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
					rectangle(frame_l, p1_l, p2_l, Scalar(0, 255, 0), 2);//���ο��ROI����


					//��ȡROI���ο���������    ((p2.x-p1.x)/2,(p2.y-p1.y)/2),   ���ں���Ŀ���������

					x_l = (p2_l.x - p1_l.x) / 2 + p1_l.x;
					y_l = (p2_l.y - p1_l.y) / 2 + p1_l.y;
					if (x_l >= frameW_l / 2)
					{
						if (y_l <= frameH_l / 2)//��һ����
						{
							x_l_ = (x_l - frameW_l / 2);
							y_l_ = (frameH_l / 2 - y_l);
						}
						else //��������
						{
							x_l_ = (x_l - frameW_l / 2);
							y_l_ = -(y_l - frameH_l / 2);
						}
					}
					else
					{
						if (y_l <= frameH_l / 2)//�ڶ�����
						{
							x_l_ = -(frameW_l / 2 - x_l);
							y_l_ = (frameH_l / 2 - y_l);
						}
						else //��������
						{
							x_l_ = -(frameW_l / 2 - x_l);
							y_l_ = -(y_l - frameH_l / 2);
						}
					}

					//�ռ������ȡ



					//��ȡʱ�������ӡ������Ϣ
					/*time_stamp();
					cout << "����ͼ������ϵ��Ŀ�����꣺" << "x_l:" << x_l << "y_l:" << y_l << endl;*/

				}

			}
			for (int i = 0; i < contours_r.size(); i++)
			{
				x_r = 0, y_r = 0;
				x_r_ = 0, y_r_ = 0;
				//contours[i]������ǵ�i��������contours[i].size()������ǵ�i�����������е����ص���,�����������ص����

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
					rectangle(frame_r, p1_r, p2_r, Scalar(0, 255, 0), 2);//���ο��ROI����


					//��ȡROI���ο���������    ((p2.x-p1.x)/2,(p2.y-p1.y)/2),   ���ں���Ŀ���������

					x_r = (p2_r.x - p1_r.x) / 2 + p1_r.x;
					y_r = (p2_r.y - p1_r.y) / 2 + p1_r.y;
					if (x_r >= frameW_r / 2)
					{
						if (y_r <= frameH_r / 2)//��һ����
						{
							x_r_ = (x_r - frameW_r / 2);
							y_r_ = (frameH_r / 2 - y_r);
						}
						else //��������
						{
							x_r_ = (x_r - frameW_r / 2);
							y_r_ = -(y_r - frameH_r / 2);
						}
					}
					else
					{
						if (y_r <= frameH_r / 2)//�ڶ�����
						{
							x_r_ = -(frameW_r / 2 - x_r);
							y_r_ = (frameH_r / 2 - y_r);
						}
						else //��������
						{
							x_r_ = -(frameW_r / 2 - x_r);
							y_r_ = -(y_r - frameH_r / 2);
						}
					}

					//�ռ������ȡ



					//��ȡʱ�������ӡ������Ϣ
					if (x_l != 0 && y_l != 0 && x_r != 0 && y_r != 0)
					{

						//����˫Ŀ�Ӿ���ѧ����ģ��
						//����Ŀ����������������ϵ�µ���ά���꣨X,Y,Z��
						Point3f worldPoint;
						worldPoint = uv2xyz(Point2f(x_l, y_l), Point2f(x_r, y_r));

						X = worldPoint.x / 1000;
						Y = - worldPoint.y / 1000;
						Z = worldPoint.z / 1000;




						paint(X / 15, Y / 15, Z / 15);//���������
						/*glutMainLoop();*/

						time_stamp();
						cout << "����ͼ������ϵ��Ŀ�����꣺" << "x_l:" << x_l_ << " y_l:" << y_l_ << endl;
						cout << "����ͼ������ϵ��Ŀ�����꣺" << "x_r:" << x_r_ << " y_r:" << y_r_ << endl;
						cout << "==*-*-*-*-*-*-*-*-*-*-*-*-*-*-*==" << endl;
						cout << "�������ϵ����ά���꣺" << "X:" << X << "m" << " Y:" << Y << "m" << " Z:" << Z << "m" << endl;
					}
				}

			}
		}
		namedWindow("���۸���ʶ�������", WINDOW_AUTOSIZE);
		imshow("���۸���ʶ�������", frame_l);
		namedWindow("���۸���ʶ�������", WINDOW_AUTOSIZE);
		imshow("���۸���ʶ�������", frame_r);

		if (cvWaitKey(33) >= 0)
			break;


	}
	glutMainLoop();
	return 0;
}








