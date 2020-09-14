#include<opencv.hpp>
#include <iostream>  
//#include <stdio.h>  
using namespace std;
using namespace cv;
Mat img(500, 500, CV_8UC3);
//������Դ��ڵ�����ֵ����Ϊ����ԭ�������Ͻǣ�����sinǰ�и�����  
static inline Point calcPoint(Point2f center, double R, double angle)
{
	return center + Point2f((float)cos(angle), (float)-sin(angle))*(float)R;
}
void drawCross(Point center, Scalar color, int d)
{
	line(img, Point(center.x - d, center.y - d),
		Point(center.x + d, center.y + d), color, 1, CV_AA, 0);
	line(img, Point(center.x + d, center.y - d),
		Point(center.x - d, center.y + d), color, 1, CV_AA, 0);
}
static void help()
{
	printf("\nExamle of c calls to OpenCV's Kalman filter.\n"
		"   Tracking of rotating point.\n"
		"   Rotation speed is constant.\n"
		"   Both state and measurements vectors are 1D (a point angle),\n"
		"   Measurement is the real point angle + gaussian noise.\n"
		"   The real and the estimated points are connected with yellow line segment,\n"
		"   the real and the measured points are connected with red line segment.\n"
		"   (if Kalman filter works correctly,\n"
		"    the yellow segment should be shorter than the red one).\n"
		"\n"
		"   Pressing any key (except ESC) will reset the tracking with a different speed.\n"
		"   Pressing ESC will stop the program.\n"
	);
}

int main(int, char**)
{
	help();

	KalmanFilter KF(2, 1, 0);                                    //�����������˲�������KF  
	Mat state(2, 1, CV_32F);                                     //state(�Ƕȣ����Ƕ�)  
	Mat processNoise(2, 1, CV_32F);
	Mat measurement = Mat::zeros(1, 1, CV_32F);                 //�������ֵ  
	char code = (char)-1;
	Scalar color;
	int d = 5;

	for (;;)
	{
		//1.��ʼ��  
		randn(state, Scalar::all(0), Scalar::all(0.1));          //  
		KF.transitionMatrix = (Mat_<float>(2, 2) << 1, 1, 0, 1);  //ת�ƾ���A[1,1;0,1]      


																   //�����漸����������Ϊ�Խ���  
		setIdentity(KF.measurementMatrix);                             //��������H  
		setIdentity(KF.processNoiseCov, Scalar::all(1e-5));            //ϵͳ�����������Q  
		setIdentity(KF.measurementNoiseCov, Scalar::all(1e-1));        //���������������R  
		setIdentity(KF.errorCovPost, Scalar::all(1));                  //����������Э�������P  

		randn(KF.statePost, Scalar::all(0), Scalar::all(0.1));          //x(0)��ʼ��  

		for (;;)
		{
			Point2f center(img.cols*0.5f, img.rows*0.5f);          //centerͼ�����ĵ�  
			float R = img.cols / 3.f;                                //�뾶  
			double stateAngle = state.at<float>(0);                //���ٵ�Ƕ�  
			Point statePt = calcPoint(center, R, stateAngle);     //���ٵ�����statePt  

																  //2. Ԥ��  
			Mat prediction = KF.predict();                       //����Ԥ��ֵ������x'  
			double predictAngle = prediction.at<float>(0);          //Ԥ���ĽǶ�  
			Point predictPt = calcPoint(center, R, predictAngle);   //Ԥ�������predictPt  


																	//3.����  
																	//measurement�ǲ���ֵ  
			randn(measurement, Scalar::all(0), Scalar::all(KF.measurementNoiseCov.at<float>(0)));     //��measurement��ֵN(0,R)�����ֵ  

																									  // generate measurement  
			measurement += KF.measurementMatrix*state;  //z = z + H*x;  

			double measAngle = measurement.at<float>(0);
			Point measPt = calcPoint(center, R, measAngle);

			// plot points  
			//�����˻�ʮ�ֵķ�����ֵ��ѧϰ��  



			img = Scalar::all(0);
			drawCross(statePt, Scalar(255, 255, 255), 3);
			drawCross(measPt, Scalar(0, 0, 255), 3);
			drawCross(predictPt, Scalar(0, 255, 0), 3);
			line(img, statePt, measPt, Scalar(0, 0, 255), 3, CV_AA, 0);
			line(img, statePt, predictPt, Scalar(0, 255, 255), 3, CV_AA, 0);


			//����kalman������correct�����õ�����۲�ֵУ�����״̬����ֵ����  
			if (theRNG().uniform(0, 4) != 0)
				KF.correct(measurement);

			//���������Ļ���������Բ���˶������˵�������������Բ���˶�����Ϊ������ԭ���˶�������ܻ�ı�  
			randn(processNoise, Scalar::all(0), Scalar::all(sqrt(KF.processNoiseCov.at<float>(0, 0))));   //vk  
			state = KF.transitionMatrix*state + processNoise;

			imshow("Kalman", img);
			code = (char)waitKey(100);

			if (code > 0)
				break;
		}
		if (code == 27 || code == 'q' || code == 'Q')
			break;
	}

	return 0;
}
