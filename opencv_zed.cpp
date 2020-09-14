//
//#include <iostream>
//#include <opencv2\opencv.hpp>
//#include<opencv.hpp>
//
//#include <sl/Camera.hpp>
//#include <windows.h>
//using namespace std;
//using namespace cv;
//using namespace sl;
//
////cv::Mat slMat2cvMat(sl::Mat& input) {
////	// Mapping between MAT_TYPE and CV_TYPE
////	int cv_type = -1;
////	switch (input.getDataType()) {
////	case MAT_TYPE_32F_C1: cv_type = CV_32FC1; break;
////	case MAT_TYPE_32F_C2: cv_type = CV_32FC2; break;
////	case MAT_TYPE_32F_C3: cv_type = CV_32FC3; break;
////	case MAT_TYPE_32F_C4: cv_type = CV_32FC4; break;
////	case MAT_TYPE_8U_C1: cv_type = CV_8UC1; break;
////	case MAT_TYPE_8U_C2: cv_type = CV_8UC2; break;
////	case MAT_TYPE_8U_C3: cv_type = CV_8UC3; break;
////	case MAT_TYPE_8U_C4: cv_type = CV_8UC4; break;
////	default: break;
////	}
////	// Since cv::Mat data requires a uchar* pointer, we get the uchar1 pointer from sl::Mat (getPtr<T>())
////	// cv::Mat and sl::Mat will share a single memory structure
////	return cv::Mat(input.getHeight(), input.getWidth(), cv_type, input.getPtr<sl::uchar1>(MEM_CPU));
////}
////int main(int argc, char** argv)
////{
////	VideoCapture cap;
////	if (argc > 1)
////		cap.open(argv[1]);
////	else
////		cap.open(0);
////
////	if (!cap.isOpened())
////	{
////		std::cerr << "Cannot read video. Try moving video file to sample directory." << std::endl;
////		return -1;
////	}
////	sl::Mat image;
////	cv::Mat frame = slMat2cvMat(image);
////	for (;;)
////	{
////		cap >> frame;
////		if (frame.empty())
////			break;
////
////		imshow("FG Segmentation", frame);
////
////
////		int c = waitKey(30);
////		if (c == 'q' || c == 'Q' || (c & 255) == 27)
////			break;
////	}
////	return 0;
////}
//
//
//cv::Mat slMat2cvMat(sl::Mat& input) {
//	// Mapping between MAT_TYPE and CV_TYPE
//	int cv_type = -1;
//	switch (input.getDataType()) {
//	case MAT_TYPE_32F_C1: cv_type = CV_32FC1; break;
//	case MAT_TYPE_32F_C2: cv_type = CV_32FC2; break;
//	case MAT_TYPE_32F_C3: cv_type = CV_32FC3; break;
//	case MAT_TYPE_32F_C4: cv_type = CV_32FC4; break;
//	case MAT_TYPE_8U_C1: cv_type = CV_8UC1; break;
//	case MAT_TYPE_8U_C2: cv_type = CV_8UC2; break;
//	case MAT_TYPE_8U_C3: cv_type = CV_8UC3; break;
//	case MAT_TYPE_8U_C4: cv_type = CV_8UC4; break;
//	default: break;
//	}
//	// Since cv::Mat data requires a uchar* pointer, we get the uchar1 pointer from sl::Mat (getPtr<T>())
//	// cv::Mat and sl::Mat will share a single memory structure
//	return cv::Mat(input.getHeight(), input.getWidth(), cv_type, input.getPtr<sl::uchar1>(MEM_CPU));
//}
//
//	
//	int main(int argc, char **argv) {
//
//		// Create a ZED camera object
//		Camera zed;
//
//		// Set configuration parameters
//		InitParameters init_params;
//		init_params.camera_resolution = RESOLUTION_HD720;
//		init_params.depth_mode = DEPTH_MODE_PERFORMANCE;
//		init_params.coordinate_units = UNIT_METER;
//		if (argc > 1) init_params.svo_input_filename.set(argv[1]);
//
//		// Open the camera
//		ERROR_CODE err = zed.open(init_params);
//		cout << "open it" << endl;
//		//if (err != SUCCESS) {
//		//	printf("%s\n", toString(err).c_str());
//		//	zed.close();
//		//	return 1; // Quit if an error occurred
//		//}
//
//
//		// Set runtime parameters after opening the camera
//		//RuntimeParameters runtime_parameters;
//		//runtime_parameters.sensing_mode = SENSING_MODE_STANDARD;
//
//		// Prepare new image size to retrieve half-resolution images
//		Resolution image_size = zed.getResolution();
//		int new_width = image_size.width / 2;
//		int new_height = image_size.height / 2;
//
//		// To share data between sl::Mat and cv::Mat, use slMat2cvMat()
//		// Only the headers and pointer to the sl::Mat are copied, not the data itself
//		sl::Mat image_zed(new_width, new_height, MAT_TYPE_8U_C4);
//		cv::Mat image_ocv = slMat2cvMat(image_zed);
//		sl::Mat depth_image_zed(new_width, new_height, MAT_TYPE_8U_C4);
//		cv::Mat depth_image_ocv = slMat2cvMat(depth_image_zed);
//		/*sl::Mat point_cloud;*/
//
//		// Loop until 'q' is pressed
//		//char key = ' ';
//		//Sleep(3000);
//		//while (key != 'q') {
//		while(1){
//
//			if (zed.grab() == SUCCESS) {
//
//				// Retrieve the left image, depth image in half-resolution
//				zed.retrieveImage(image_zed, VIEW_LEFT, MEM_CPU, new_width, new_height);
//				zed.retrieveImage(depth_image_zed, VIEW_DEPTH, MEM_CPU, new_width, new_height);
//
//				// Retrieve the RGBA point cloud in half-resolution
//				// To learn how to manipulate and display point clouds, see Depth Sensing sample
//				//zed.retrieveMeasure(point_cloud, MEASURE_XYZRGBA, MEM_CPU, new_width, new_height);
//
//				// Display image and depth using cv:Mat which share sl:Mat data
//				cv::imshow("Image", image_ocv);
//				cv::imshow("Depth", depth_image_ocv);
//
//				// Handle key event
//				//key = cv::waitKey(10);
//				//processKeyEvent(zed, key);
//				if (waitKey(30) >= 0) break;
//			}
//		}
//		//zed.close();
//		return 0;
//	}
//
//
//
//
//
////
////int main(int argc, char **argv) {
////
////	// Create a ZED camera object
////	Camera zed;
////	/*Camera* zed=new Camera();*/
////	// Set configuration parameters
////	InitParameters init_params;
////	init_params.depth_mode = DEPTH_MODE_PERFORMANCE; // Use PERFORMANCE depth mode
////	init_params.coordinate_units = UNIT_MILLIMETER; // Use millimeter units (for depth measurements)
////	
////	// Open the camera
////	ERROR_CODE err = zed.open(init_params);
////	if (err != SUCCESS)
////		exit(-1);
////	
////	// Set runtime parameters after opening the camera
////	RuntimeParameters runtime_parameters;
////	runtime_parameters.sensing_mode = SENSING_MODE_STANDARD; // Use STANDARD sensing mode
////
////	// Capture 50 images and depth, then stop
////	//int j = 0;
////	//sl::Mat point_cloud,leftimage,depthimage,cloudimage;
////
////	int width = zed.getResolution().width;
////	int height = zed.getResolution().height;
////	sl::Mat image(width, height, MAT_TYPE_8U_C4);
////	sl::Mat depth(width, height, MAT_TYPE_8U_C4);
////
////	cv::Mat image_cv = slMat2cvMat(image);
////	cv::Mat depyh_cv = slMat2cvMat(depth);
////
////	 //Create OpenCV windows
////	//cv::namedWindow("Image", cv::WINDOW_AUTOSIZE);
////	//cv::namedWindow("Depth", cv::WINDOW_AUTOSIZE);
////	
////
////	// Settings for windows
////	cv::Size displaySize(720, 404);
////	/*cv::Mat imageDisplay(displaySize, CV_8UC4);
////	cv::Mat depthDisplay(displaySize, CV_8UC4);*/
////
////
////
////	while (1) {
////		// A new image is available if grab() returns SUCCESS
////		if (zed.grab(runtime_parameters) == SUCCESS) 
////		{
////			// Retrieve left image
////			zed.retrieveImage(image, VIEW_LEFT);
////			//memcpy(image.data, left.data, width*height * 4 * sizeof(uchar));
////			cv::resize(zed.retrieveImage(image, VIEW_LEFT), image_cv, displaySize);
////            cv::imshow("Image", image_cv);
////
////			// Retrieve depth map. Depth is aligned on the left image
////
////
////
////			/*zed.retrieveImage(depth, VIEW_DEPTH);
////			cv::resize(zed.retrieveMeasure(depth, MEASURE_DEPTH), depyh_cv, displaySize);
////			cv::imshow("Depth", depyh_cv);*/
////
////
////
////			cv::waitKey(30);
////		}
////	}
////	// Close the camera
////	zed.close();
////	return 0;
////}
//
//


//
//#include<gl/GLUT.H>
//void Initial(void)//��ʼ������ 
//{
//	glClearColor(1.0f, 1.0f, 1.0f, 1.0f);//��ɫ������ǰ3����RGB�������Alphaֵ����������͸����1.0��ʾ��ȫ��͸��
//	glMatrixMode(GL_PROJECTION);//OpenGL������ά��ʽ������ͼ��������Ҫһ��ͶӰ�任����άͼ��ͶӰ����ʾ���Ķ�ά�ռ���
//	gluOrtho2D(0.0, 200, 0.0, 150.0);//ָ��ʹ����ͶӰ��һ��x������0~200��y����0~150��Χ�ڵľ�����������ͶӰ����ʾ������
//
//}
//void myDisplay(void)//��ʾ�ص�����
//{
//	glClear(GL_COLOR_BUFFER_BIT);//ʹ��glClearColorz��ָ����ֵ�趨��ɫ��������ֵ�����������е�ÿһ����������Ϊ����ɫ
//	glColor3f(0.0f, 0.0f, 0.0f);//��ͼ��ɫΪ��ɫ
//	glRectf(50.0f, 100.0f, 150.0f, 50.0f);//ͼ�ε����꣬����һ�����Ͻ��ڣ�50��100�������½��ڣ�150��50���ľ���
//	glFlush();//���OpenGL���������ǿ��ִ���������������OpenGL����
//}
//int main(int argc, char * argv[])//����ʹ��glut�⺯�����д��ڹ���
//{
//	glutInit(&argc, argv);//ʹ��glut����Ҫ���г�ʼ��
//	glutInitDisplayMode(GLUT_RGB | GLUT_SINGLE);//�趨������ʾģʽ����ɫģ�ͺͻ��棬������RGB��ɫģ�ͺ͵�����
//	glutInitWindowPosition(100, 100);//�趨���ڵĳ�ʼλ�ã���Ļ���Ͻ�Ϊԭ�㣬��λΪ����
//	glutInitWindowSize(400, 400);//�趨���ڵĴ�С
//	glutCreateWindow("��һ��OpenGL���򡪡���ΰ");//����һ�����ڣ������Ǵ��ڱ�����
//	glutDisplayFunc(&myDisplay);//��myDisplayָ��Ϊ��ǰ���ڵ���ʾ���ݺ���
//	Initial();
//	glutMainLoop();//ʹ���ڿ������������ʹ��ʾ�ص�������ʼ����
//	return 0;
//}


#include<windows.h>
#include <GL/glut.h>
#pragma comment(linker,"/subsystem:\"windows\" /entry:\"mainCRTStartup\"")

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

//������һ��������ĺ���
void axis(double length)
{
	glColor3f(1.0f, 0.0f, 0.0f);
	glPushMatrix();
	//++++++++++++++++++++++++++++++++++++++++
	glLineWidth(3.f);
	
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
	//++++++++++++++++++++++++++++++++++
	glClearColor(0, 0, 0, 1);

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
	


	glColor3f(0.0f, 1.0f, 0.0f);
	//glClear(GL_COLOR_BUFFER_BIT);
	glRasterPos3d(0, 0, -2.18);
	drawString("Z");
	glRasterPos3d(0, -2.18, 0);
	drawString("Y");
	glRasterPos3d(2.18, 0, 0);
	drawString("X");
	glRasterPos3d(0, -0.12, 0);
	drawString("O");
	//glPushMatrix();
	//glBegin(GL_LINES);
	//glVertex3d(m[0][0],m[0][1],m[0][2]);
	//glVertex3d(m[1][0], m[1][1], m[1][2]);//�Ȼ�һ�ᣬZ��
	//glEnd();
	//glPopMatrix();

	//Sleep(2000);

	GLfloat pointSize = 18.0f;
	glPointSize(pointSize);



	//+++++++++++++++++++++++++++++++++++++++++++
	glColor3f(0.0f, 0.0f, 1.0f);




	glBegin(GL_POINTS);
	/*for (i = 0; i < 2; i++)
	{*/
		/*GLfloat pointSize = 15.0f;
		glPointSize(pointSize);
		glBegin(GL_POINTS);*/
		glVertex3d(x, -y, -z);
		/*glEnd();*/
		//Sleep(1000);
	//}
	glEnd();
	
	/*GLfloat pointSize = 15.0f;
	glPointSize(pointSize);
	glBegin(GL_POINTS);
	glVertex3d(m[0][0], -m[0][1], -m[0][2]);
	glEnd();*/

	glutSwapBuffers();
	
}
void Init()
{
	glClearColor(1.0, 1.0, 1.0, 1.0);
}


int main(int argv, char *argc[])
{
	
	
	double x=0,y=0,z=0;
	glutInit(&argv, argc);
	glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE);
	glutInitWindowSize(1000, 1000);
	glutInitWindowPosition(1000, 50);
	glutCreateWindow("3D�ռ�Ŀ���˶�����");
	Init();

	paint(x,y,z);


	glutMainLoop();
}