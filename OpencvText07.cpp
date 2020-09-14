//֡�Ŀ��ʶ��
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

int main(int argc, char** argv)
{
	VideoCapture capture("F:\\visual c++&&opencv\\source\\uav.mp4");
	if (!capture.isOpened())
	{
		cout << "No camera or video input!\n" << endl;
		return -1;
	}
	Mat frame;//mat������Iplimag
	Mat gray;
	Mat gray_dilate1;
	Mat gray_dilate2;
	Mat gray_dilate3;
	Mat background, foreground, foreground_BW;
	Mat mid_filer;   //��ֵ�˲��������Ƭ

	const char* pzRotatingWindowName = "Rotated Video";
	namedWindow(pzRotatingWindowName, CV_WINDOW_AUTOSIZE);

	/*int iAngle = 180;
	createTrackbar("Angle", pzRotatingWindowName, &iAngle, 360);*/

	//---------------------------------------------------------------------
	//��ȡ��Ƶ�Ŀ�ȡ��߶ȡ�֡�ʡ��ܵ�֡��
	int frameH = capture.get(CV_CAP_PROP_FRAME_HEIGHT); //��ȡ֡��
	int frameW = capture.get(CV_CAP_PROP_FRAME_WIDTH);  //��ȡ֡��
	int fps = capture.get(CV_CAP_PROP_FPS);          //��ȡ֡��
	int numFrames = capture.get(CV_CAP_PROP_FRAME_COUNT);  //��ȡ����֡��
	int num = numFrames;
	printf("vedio's \nwidth = %d\t height = %d\n video's fps = %d\t nums = %d", frameW, frameH, fps, numFrames);
	//---------------------------------------------------------------------
	Mat frame_0, frame_1;//Mat m(3, 5, CV_32FC1, 1);
	//---------------------------------------------------------------------
	while (1)
	{
		capture >> frame;
		Mat matRotation = getRotationMatrix2D(Point(frame.cols / 2, frame.rows / 2), 270, 1);
		// Rotate the image
		Mat matRotatedFrame;
		warpAffine(frame, matRotatedFrame, matRotation, frame.size());
		//apture >> frame;
		imshow(pzRotatingWindowName, matRotatedFrame);
		cvtColor(matRotatedFrame, gray, CV_RGB2GRAY);
		//-----------------------------------------------------------------------------------
		//ѡ��ǰһ֡��Ϊ�����������һ֡ʱ����һ֡��Ϊ������
		if (num == numFrames)
		{
			background = gray.clone();
			frame_0 = background;
		}
		else
		{
			background = frame_0;
		}
		//------------------------------------------------------------------------------------
		absdiff(gray, background, foreground);//��֡���ǰ��
		imshow("foreground", foreground);
		Mat element = getStructuringElement(MORPH_RECT, Size(5, 5));


		//threshold(foreground, foreground_BW, 30, 255, 0);//��ֵ��ͨ������Ϊ50  255
		threshold(foreground, foreground_BW, 0, 255 ,CV_THRESH_BINARY | CV_THRESH_OTSU) ;  //�˴�ʹ�ô��  ����Ӧȡ��ֵ
		//imshow("foreground_BW", foreground_BW);
		//medianBlur(foreground_BW, mid_filer, 3);     //��ֵ�˲���
		//imshow("mid_filer", mid_filer);
		dilate(foreground_BW, gray_dilate1, element);
		//imshow("gray_dilate1", gray_dilate1);
		dilate(gray_dilate1, gray_dilate2, element);
		//imshow("gray_dilate2", gray_dilate2);
		dilate(gray_dilate2, gray_dilate3, element);
		//imshow("gray_dilate3", gray_dilate3);
		medianBlur(gray_dilate3, mid_filer, 3);     //��ֵ�˲���
		imshow("mid_filer", mid_filer);
		frame_0 = gray.clone();
		num--;
		char c = waitKey(33);
		if (c == 27) break;
		if (num < 1)
			return -1;
	}
}
