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
		inRange(frame, Scalar(0, 127, 0), Scalar(120, 255, 120), mask); // 过滤，图像二值化
		morphologyEx(mask, mask, MORPH_OPEN, kernel1, Point(-1, -1), 1); // 开操作，去椒盐噪声
		dilate(mask, mask, kernel2, Point(-1, -1), 4);// 膨胀，为后续的发现最大外接轮廓做预处理
		imshow("track mask", mask);

		processFrame(mask, roi); // 轮廓发现与在frame上位置标定
		rectangle(frame, roi, Scalar(0, 0, 255), 3, 8, 0); // 绘制位置标定
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
			double area = contourArea(contours[static_cast<int>(t)]); // 迭代找到最大外接矩形
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