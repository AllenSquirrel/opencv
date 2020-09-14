//������
#include "opencv2/video/tracking.hpp"  
#include "opencv2/highgui/highgui.hpp"  
#include <stdio.h>  
#include<iostream>
using namespace cv;
using namespace std;

const int winHeight = 600;
const int winWidth = 800;


Point mousePosition = Point(winWidth >> 1, winHeight >> 1);

//mouse event callback  
void mouseEvent(int event, int x, int y, int flags, void *param)
{
	if (event == CV_EVENT_MOUSEMOVE) {
		mousePosition = Point(x, y);
	}
}

int main(void)
{
	Mat processNoise(2, 1, CV_32F);//�¼ӳ���
	RNG rng;
	//1.kalman filter setup  
	const int stateNum = 4;                                      //״ֵ̬4��1����(x,y,��x,��y)  
	const int measureNum = 2;                                    //����ֵ2��1����(x,y)    
	KalmanFilter KF(stateNum, measureNum, 0);

	KF.transitionMatrix = (Mat_<float>(4, 4) << 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1);  //ת�ƾ���A  
	setIdentity(KF.measurementMatrix);                                             //��������H  
	setIdentity(KF.processNoiseCov, Scalar::all(5));                            //ϵͳ�����������Q  
	//cout << KF.processNoiseCov;
	//waitKey(0);
	setIdentity(KF.measurementNoiseCov, Scalar::all(1e-1));                        //���������������R  
	setIdentity(KF.errorCovPost, Scalar::all(1));                                  //����������Э�������P  
	rng.fill(KF.statePost, RNG::UNIFORM, 0, winHeight > winWidth ? winWidth : winHeight);   //��ʼ״ֵ̬x(0)  
	Mat measurement = Mat::zeros(measureNum, 1, CV_32F);                           //��ʼ����ֵx'(0)����Ϊ����Ҫ�������ֵ�����Ա����ȶ���  

	namedWindow("kalman");
	setMouseCallback("kalman", mouseEvent);

	Mat image(winHeight, winWidth, CV_8UC3, Scalar(0));

	while (1)
	{
		//2.kalman prediction  
		Mat prediction = KF.predict();
		Point predict_pt = Point(prediction.at<float>(0), prediction.at<float>(1));   //Ԥ��ֵ(x',y')  

																					  //3.update measurement  

		randn(processNoise, Scalar::all(0), Scalar::all(sqrt(KF.processNoiseCov.at<float>(0, 0))));//�¼ӳ���
	   // cout << processNoise;//�¼ӳ���
		//waitKey(0);//�¼ӳ���
		//measurement.at<float>(0) = (float)mousePosition.x;
		//measurement.at<float>(1) = (float)mousePosition.y;
		//Point predict_pt = Point(prediction.at<float>(0), prediction.at<float>(1));
		measurement.at<float>(0) = (float)(mousePosition.x + processNoise.at<float>(0, 0));
		measurement.at<float>(1) = (float)(mousePosition.y + processNoise.at<float>(1, 0));
		//cout << measurement.at<float>(0);
		//waitKey(0);
		Point p;
		p.x = measurement.at<float>(0);
		p.y = measurement.at<float>(1);
		//4.update  
		KF.correct(measurement);

		//draw   
		image.setTo(Scalar(255, 255, 255, 0));
		circle(image, predict_pt, 5, Scalar(0, 255, 0), 3);    //predicted point with green  
		circle(image, mousePosition, 5, Scalar(255, 0, 0), 3); //current position with red          

		char buf[256];
		sprintf_s(buf, 256, "predicted position:(%3d,%3d)", predict_pt.x, predict_pt.y);
		putText(image, buf, Point(10, 30), CV_FONT_HERSHEY_SCRIPT_COMPLEX, 1, Scalar(0, 0, 0), 1, 8);
		sprintf_s(buf, 256, "measure position :(%3d,%3d)", p.x, p.y);
		putText(image, buf, cvPoint(10, 60), CV_FONT_HERSHEY_SCRIPT_COMPLEX, 1, Scalar(0, 0, 0), 1, 8);
		sprintf_s(buf, 256, "current position :(%3d,%3d)", mousePosition.x, mousePosition.y);
		putText(image, buf, cvPoint(10, 90), CV_FONT_HERSHEY_SCRIPT_COMPLEX, 1, Scalar(0, 0, 0), 1, 8);

		imshow("kalman", image);
		int key = waitKey(3);
		if (key == 27) {//esc     
			break;
		}
	}
}
