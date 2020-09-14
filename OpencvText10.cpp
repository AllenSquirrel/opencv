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

#define R {0.01297,0.02971,0.00340}  //�궨��ת����
#define T {-404.97875;-87.07289;272.21939}  //�궨ƽ�ƾ���


//******************************************************************





int time_stamp()//ʱ�������,��ȡ��ǰϵͳʱ��
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






vector<Point> match_feature1()    //������״����ƥ��
{
	Mat img_gray;
	Mat mid_filer2;
	
	Mat img = imread("F:\\visual c++&&opencv\\source\\1.png", CV_LOAD_IMAGE_UNCHANGED);

	Mat draw_img = img.clone();
	//namedWindow("jpg", WINDOW_AUTOSIZE);
	//imshow("jpg",img);

	cvtColor(img, img_gray, CV_BGR2GRAY);
	//Sobel(img_gray, img_gray, CV_8U, 1, 0, 3, 0.4, 128);
	//threshold(img_gray, img_gray,1, 255, CV_THRESH_BINARY);//ȷ�������Ұ�
	adaptiveThreshold(img_gray, img_gray,255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 25, 5);//����Ӧ��ֵ�ָ�


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
	return contours1[0];
}


vector<Point> match_feature2()    //������״����ƥ��
{
	Mat img_gray;
	Mat mid_filer2;

	Mat img = imread("F:\\visual c++&&opencv\\source\\2.png", CV_LOAD_IMAGE_UNCHANGED);

	Mat draw_img = img.clone();
	//namedWindow("jpg", WINDOW_AUTOSIZE);
	//imshow("jpg",img);

	cvtColor(img, img_gray, CV_BGR2GRAY);
	//Sobel(img_gray, img_gray, CV_8U, 1, 0, 3, 0.4, 128);
	//threshold(img_gray, img_gray,1, 255, CV_THRESH_BINARY);//ȷ�������Ұ�
	adaptiveThreshold(img_gray, img_gray, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 25, 5);//����Ӧ��ֵ�ָ�


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
	return contours1[0];
}







int main(int argc, unsigned char* argv[])
{
	//match_feature();

	double x_l, y_l;//ͼ������ϵ��Ŀ���άƽ������
	double X, Y, Z;//�����������ϵ��Ŀ����ά�ռ�����
	double f_l, f_r;//���۽��࣬���۽���

	f_l = (fx_l + fy_l) / 2;
	f_r = (fx_r + fy_r) / 2;


	Mat frame;
	Mat img_src1, img_src2, img_src3;//3֡����Ҫ3֡ͼƬ
	Mat img_dst, gray1, gray2, gray3;
	Mat gray_diff1, gray_diff2;//�洢2�������ͼƬ
	Mat gray_diff11, gray_diff12;
	Mat gray_diff21, gray_diff22;
	Mat gray;//������ʾǰ����
	Mat mid_filer;   //��ֵ�˲��������Ƭ
	bool pause = false;


	VideoCapture vido_file("F:\\visual c++&&opencv\\source\\008.mp4");//���������Ӧ���ļ���
	namedWindow("foreground", WINDOW_NORMAL);

	//---------------------------------------------------------------------
	//��ȡ��Ƶ�Ŀ�ȡ��߶ȡ�֡�ʡ��ܵ�֡��
	int frameH = vido_file.get(CV_CAP_PROP_FRAME_HEIGHT); //��ȡ֡��
	int frameW = vido_file.get(CV_CAP_PROP_FRAME_WIDTH);  //��ȡ֡��
	int fps = vido_file.get(CV_CAP_PROP_FPS);          //��ȡ֡��
	int numFrames = vido_file.get(CV_CAP_PROP_FRAME_COUNT);  //��ȡ����֡��
	int num = numFrames;
	printf("video's \nwidth = %d\t height = %d\n video's FPS = %d\t nums = %d\n", frameW, frameH, fps, numFrames);
	//---------------------------------------------------------------------

	while(1)
	{
		

		vido_file >>frame;
		Mat matRotation = getRotationMatrix2D(Point(frame.cols / 2, frame.rows / 2), 270, 1);//��ȡͼ�����ĵ���ת����
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
				printf("��ȡ֡ʧ��");
				break;
			}
			cvtColor(img_src1, gray1, CV_BGR2GRAY);

			waitKey(3);//���ǵ�pc�������ٶȣ�ÿ��33ms��ȡһ֡ͼ�񣬲�����ת��Ϊ�Ҷ�ͼ��ֱ���
			vido_file >> img_src2;
			if (&img_src2 == nullptr) 
			{
				printf("��ȡ֡ʧ��");
				break;
			}
			cvtColor(img_src2, gray2, CV_BGR2GRAY);

			waitKey(3);
			vido_file >> img_src3;
			if (&img_src3 == nullptr) //��Ҫ�ж���Ƶ����ʱ����ȡ֡ʧ�ܵ����
			{
				printf("�������");
				break;
			}
			cvtColor(img_src3, gray3, CV_BGR2GRAY);


			Sobel(gray1, gray1, CV_8U, 1, 0, 3, 0.4, 128);//sobel���Ӽ�����ͼ���֣�����sobel���ӽ����Gaussianƽ����΢�֣����ԣ����������ٶ�������һ��³����
			Sobel(gray2, gray2, CV_8U, 1, 0, 3, 0.4, 128);
			Sobel(gray3, gray3, CV_8U, 1, 0, 3, 0.4, 128);



			subtract(gray2, gray1, gray_diff11);//�ڶ�֡����һ֡
			subtract(gray1, gray2, gray_diff12);
			add(gray_diff11, gray_diff12, gray_diff1);
			subtract(gray3, gray2, gray_diff21);//����֡���ڶ�֡
			subtract(gray2, gray3, gray_diff22);
			add(gray_diff21, gray_diff22, gray_diff2);

			for (int i = 0; i < gray_diff1.rows; i++)
				for (int j = 0; j < gray_diff1.cols; j++)
				{
					if (abs(gray_diff1.at<unsigned char>(i, j)) >= threshold_diff1)//����ģ�����һ��Ҫ��unsigned char�������һֱ����
						gray_diff1.at<unsigned char>(i, j) = 255;            //��һ�������ֵ����
					else gray_diff1.at<unsigned char>(i, j) = 0;

					if (abs(gray_diff2.at<unsigned char>(i, j)) >= threshold_diff2)//�ڶ��������ֵ����
						gray_diff2.at<unsigned char>(i, j) = 255;
					else gray_diff2.at<unsigned char>(i, j) = 0;
				}
			bitwise_and(gray_diff1, gray_diff2, gray);//��֡������������ͼ�����������

			dilate(gray, gray, Mat()); erode(gray, gray, Mat());//���ͺ͸�ʴ������Ч������������

			medianBlur(gray, mid_filer, 3);//��ֵ�˲�
		
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
			findContours(matRotatedFrame1, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, Point());//Ѱ������
			
			Rect bt_point;//rect��������������ζ�������ϽǺ����½����꣨ͼ������ϵ�£�
			Point p1, p2;
	        Mat imageContours = Mat::zeros(matRotatedFrame1.size(), CV_8UC1);
			Mat Contours = Mat::zeros(matRotatedFrame1.size(), CV_8UC1);  //����  


			line(matRotatedFrame, Point(frameW/2-160, frameH/2), Point(frameW / 2 + 160, frameH/2), Scalar(0, 0, 255), 3, 8);
			line(matRotatedFrame, Point(frameW / 2, frameH / 2-160), Point(frameW / 2, frameH / 2 + 160), Scalar(255, 0, 0), 3, 8);

			for (int i = 0; i < contours.size(); i++)
			{
				//contours[i]������ǵ�i��������contours[i].size()������ǵ�i�����������е����ص���,�����������ص����

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
					rectangle(matRotatedFrame, p1, p2, Scalar(0, 255, 0), 2);//���ο��ROI����


					//��ȡROI���ο���������    ((p2.x-p1.x)/2,(p2.y-p1.y)/2),   ���ں���Ŀ���������

					x_l = (p2.x - p1.x) / 2 + p1.x;
					y_l = (p2.y - p1.y) / 2 + p1.y;
					if (x_l >= frameW / 2)
					{
						if (y_l <= frameH / 2)//��һ����
						{
							x_l = (x_l - frameW / 2);
							y_l = (frameH / 2 - y_l);
						}
						else //��������
						{
							x_l = (x_l - frameW / 2);
							y_l = -(y_l - frameH /2);
						}
					}
					else
					{
						if (y_l <= frameH / 2)//�ڶ�����
						{
							x_l = -(frameW / 2 - x_l);
							y_l = (frameH / 2 - y_l);
						}
						else //��������
						{
							x_l = -(frameW / 2 - x_l);
							y_l = -(y_l - frameH / 2);
						}
					}

					//�ռ������ȡ



					//��ȡʱ�������ӡ������Ϣ
					time_stamp();
					cout << "����ͼ������ϵ��Ŀ�����꣺" << "x_l:" << x_l << "y_l:" << y_l << endl;

				}

			}
		}
		namedWindow("����ʶ�������", WINDOW_AUTOSIZE);
		imshow("����ʶ�������", matRotatedFrame);
		if (cvWaitKey(33) >= 0)
			break;
		
	}
	return 0;
}
