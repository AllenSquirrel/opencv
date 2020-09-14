#include "opencv.hpp"

static Rect roi;
static void processFrame(Mat &binary, Rect &rect);

void main(int argc, char* argv)
{
	VideoCapture capture;
	capture.open(getCVImagesPath("videos/video_006.mp4"));
	if (!capture.isOpened()) cout << "video not open.." << endl;

	Mat frame, mask;
	Mat kernel1 = getStructuringElement(MORPH_RECT, Size(3, 3), Point(-1, -1));
	Mat kernel2 = getStructuringElement(MORPH_RECT, Size(5, 5), Point(-1, -1));
	while (capture.read(frame))
	{
		inRange(frame, Scalar(0, 127, 0), Scalar(120, 255, 120), mask); // ���ˣ�ͼ���ֵ��
		morphologyEx(mask, mask, MORPH_OPEN, kernel1, Point(-1, -1), 1); // ��������ȥ��������
		dilate(mask, mask, kernel2, Point(-1, -1), 4);// ���ͣ�Ϊ�����ķ���������������Ԥ����
		imshow("track mask", mask);

		processFrame(mask, roi); // ������������frame��λ�ñ궨
		rectangle(frame, roi, Scalar(0, 0, 255), 3, 8, 0); // ����λ�ñ궨
		imshow("src6-6", frame);

		if (waitKey(100) == 27) break;
	}
	capture.release();

	waitKey(0);
}

void processFrame(Mat &binary, Rect &rect)
{
	vector<vector<Point>> contours;
	vector<Vec4i> hireachy;
	findContours(binary, contours, hireachy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, Point(0, 0));
	if (contours.size() > 0)
	{
		double maxArea = 0.0;
		for (size_t t = 0; t < contours.size(); t++)
		{
			double area = contourArea(contours[static_cast<int>(t)]); // �����ҵ������Ӿ���
			if (area > maxArea)
			{
				maxArea = area;
				rect = boundingRect(contours[static_cast<int>(t)]);
			}
		}
	}
	else
	{
		rect.x = rect.y = rect.width = rect.height = 0;
	}
}