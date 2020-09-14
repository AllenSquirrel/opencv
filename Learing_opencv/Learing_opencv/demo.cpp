#include<demo.h>
#include<opencv2/dnn.hpp> //����ʶ��demo��Ҫʹ��OpenCV�ٷ��ṩ������ģ��

using namespace std;
using namespace cv;

void imagedemo::colorspace_demo(Mat &image)
{
	Mat gray, hsv;
	cvtColor(image,hsv,COLOR_BGR2HSV);//RGBתHSV
	cvtColor(image, gray, COLOR_BGR2GRAY);//RGBת�Ҷ�
	imshow("HSV",hsv);
	imshow("GRAY", gray);
	imwrite("F:/������Ӿ�/visual c++&&opencv/source/timg_hsv.png",hsv);
}
//***********************************************************************
void imagedemo::mat_create_demo(Mat &image)
{
	Mat m1, m2,m3,m4;
	m1 = image.clone();//m1��image��Ϊͬһ��
	image.copyTo(m2);//m2��image��Ϊͬһ��
	m3 = image;//m3��imageΪͬһ��

	m4 = Mat::zeros(image.size(), image.type());  //ͨ��zeros������ʽ����ͼƬ����  ��image������ͬ��С ��ͨ��  ����ʽΪ����size��8,8����CV_8UC3��;
	std::cout << m4.cols << m4.rows <<m4.channels()<< std::endl;

}
//***********************************************************************
void imagedemo::pixel_visit_demo(Mat &image)
{
	int w = image.cols;//x
	int h = image.rows;//y
	int channel = image.channels();

	for (int col = 0; col < w; col++)
	{
		for (int row = 0; row < h; row++)
		{
			if (channel == 1)
			{
				int pix_value = image.at<uchar>(col, row);//������ȡ����
				image.at<uchar>(col, row) = 255 - pix_value;//���ظ�ֵ����
			}
			if (channel == 3)
			{
				Vec3b bgr_pix_value = image.at<Vec3b>(col, row);//������ȡ����
				image.at<Vec3b>(col, row) = 255 - bgr_pix_value[0];//���ظ�ֵ����
				image.at<Vec3b>(col, row) = 255 - bgr_pix_value[1];
				image.at<Vec3b>(col, row) = 255 - bgr_pix_value[2];
			}
		}
	}
}
//***********************************************************************
void imagedemo::pixel_op_demo(Mat &image)
{
	Mat dst = Mat::zeros(image.size(),image.type());
	Mat m = Mat::zeros(image.size(), image.type());
	m = Scalar(50,50,50);
	add(image, m, dst);//ͼƬ������������
	imshow("�ӷ�����", dst);

	subtract(image, m, dst);//ͼƬ����䰵
	imshow("��������", dst);

	multiply(image, m, dst);//ͼƬ��������Աȶ� m.scalar>1
	imshow("�˷�����", dst);

	divide(image, m, dst);//ͼƬ�����С�Աȶ�
	imshow("��������", dst);
}
//***********************************************************************
static void on_lightness(int b, void *userdata)
{
	Mat image = *((Mat*)userdata);//������Ĳ���תΪmat��������
	Mat dst = Mat::zeros(image.size(), image.type());
	Mat m = Mat::zeros(image.size(), image.type());
	addWeighted(image, 1, m, 0, b, dst);   //���ݹ�ʽ x*a+y*b+c   -->image*1+m*0+b   ������ǿ
	imshow("������Աȶȵ���",dst);
}
static void on_contrast(int b, void *userdata)
{
	Mat image = *((Mat*)userdata);//������Ĳ���תΪmat��������
	Mat dst = Mat::zeros(image.size(), image.type());
	Mat m = Mat::zeros(image.size(), image.type());
	double contrast = b / 200.0;
	addWeighted(image, contrast, m, 0.0, 0.0, dst);   //���ݹ�ʽ x*a+y*b+c   -->image*1+m*0+b   ������ǿ
	imshow("������Աȶȵ���", dst);
}
void imagedemo::tracking_bar_demo(Mat &image)
{
	namedWindow("������Աȶȵ���",WINDOW_AUTOSIZE);
	int lightness = 50;
	int maxvalue_light = 100;
	int contrast = 100;
	int maxvalue_contrast = 200;
	createTrackbar("light_bar:","������Աȶȵ���",&lightness,maxvalue_light,on_lightness,(void*)(&image));
	createTrackbar("contrast_bar:", "������Աȶȵ���", &contrast, maxvalue_contrast, on_contrast, (void*)(&image));
	on_lightness(50, &image);
	// >50 ͼƬ��������   <50 ͼƬ���ȼ�С
	// >100 ͼƬ�Աȶ�����   <100 ͼƬ�Աȶȼ�С
}
//***********************************************************************
void imagedemo::key_demo(Mat &image)
{
	Mat dst;
	while (true)
	{
		int c = waitKey(100);//100msѭ���������̲����ź�
		if (c == 27) 
		{
			std::cout << "�˳�" << std::endl;
			break;
		}
		if (c == 49) //����cֵ��С   ��ͨ�����̲��Եó�
		{
			std::cout << "enter key #1" << std::endl;
			cvtColor(image, dst, COLOR_BGR2HSV);
		}
		if (c == 50)
		{
			std::cout << "enter key #2" << std::endl;
			cvtColor(image, dst, COLOR_BGR2GRAY);
		}
		if (c == 51)
		{
			std::cout << "enter key #3" << std::endl;
		}
		imshow("������Ӧ����", dst);
	}
}
//***********************************************************************
void imagedemo::color_style_demo(Mat &image)
{
	int colormap[] = {
		COLORMAP_AUTUMN,
		COLORMAP_BONE,
		COLORMAP_COOL,
		COLORMAP_HOT,
		COLORMAP_HSV,
		COLORMAP_JET,
		COLORMAP_OCEAN,
		COLORMAP_PARULA,
		COLORMAP_PINK,
		COLORMAP_RAINBOW,
		COLORMAP_SPRING,
		COLORMAP_SUMMER,
		COLORMAP_WINTER
	};
	Mat dst;
	int index = 0;
	while (true)
	{
		int c = waitKey(2000);//2sѭ������
		if (c == 27)
		{
			break;
		}
		applyColorMap(image,dst,colormap[index%13]);
		index++;
		imshow("ͼƬ���ת��", dst);
	}
}
//***********************************************************************
void imagedemo::bitwise_demo(Mat &image)
{
	Mat m1 = Mat::zeros(Size(256,256),CV_8UC3);
	Mat m2 = Mat::zeros(Size(256, 256), CV_8UC3);
	rectangle(m1,Rect(100,100,80,80),Scalar(255,255,0),-1,LINE_8,0);//����-1��ʾ ���ʽ���ƾ���  1��ʾ�߿�ʽ���ƾ���
	rectangle(m1, Rect(150, 150, 80, 80), Scalar(0, 255, 255), -1, LINE_8, 0);

	imshow("m1", m1);
	imshow("m2", m2);

	Mat dst1,dst2,dst3,dst4;
	bitwise_and(m1, m2,dst1);
	imshow("�����",dst1);
	bitwise_or(m1, m2, dst2);
	imshow("�����", dst2);
	bitwise_xor(m1, m2, dst3);
	imshow("������", dst3);
}
//***********************************************************************
void imagedemo::channels_demo(Mat &image)
{
	std::vector<Mat> mv;
	split(image, mv);//RGBͨ������
	imshow("blue",mv[0]);//B
	imshow("gree", mv[1]);//G
	imshow("red", mv[2]);//R

	Mat dst;
	mv[1] = 0;
	mv[2] = 0;//����gree��red��ɫ,ֻ������ɫ
	merge(mv, dst);//��ͨ���ϲ�
	imshow("��ɫ",dst);

	int from_to[] = {0,2,1,1,2,0};//����1ͨ������  0,2ͨ������
	mixChannels(&image,1,&dst,1,from_to,3); //ͨ�����
	imshow("B-Rͨ�����", dst);
}
//***********************************************************************
void imagedemo::inrange_demo(Mat &image)
{
	Mat hsv, mask;
	cvtColor(image, hsv, COLOR_BGR2HSV);
	inRange(hsv,Scalar(35,43,46),Scalar(77,255,255),mask);//����ͬ��ɫ�ֿ�����ɫ��������  ��ʱǰ��ͼ��Ϊ0���ڣ�����Ϊ1��255���ף�
	imshow("mask",mask);

	Mat redback = Mat::zeros(image.size(), image.type());
	redback = Scalar(40,40,200);//��ɫ����
	bitwise_not(mask, mask); //ȡ������  ��ʱǰ��ͼ��Ϊ��  ����Ϊ0���ڣ�
	imshow("roi������ȡ",redback);
}

//***********************************************************************
void imagedemo::pixel_statistic_demo(Mat &image)
{
	double minv, maxv;
	Point minloc, maxloc;
	std::vector<Mat>mv;
	split(image,mv);//ͼ��ͨ�����в��  ��ʱRGB ��ͨ����ֿ�
	for(int i=0; i < mv.size(); i++)
	{
		minMaxLoc(mv[i],&minv,&maxv,&minloc,&maxloc,Mat());//����� ��С����ֵ
		std::cout << "No.channels:" << i << "min_value:" << minv << "max_value:" << maxv << std::endl;
	}
	Mat mean, stddev;
	meanStdDev(image, mean, stddev);//�����ؾ�ֵ ����  ÿһ��ͨ������һ����ֵ�ͷ���
	std::cout << "mean:" << mean << "stddev:" << stddev << std::endl;
	mean.at<double>(0, 0);//ȡ3*1ά�е�һ����ֵ  ����һ��ͨ�������ؾ�ֵ
	mean.at<double>(1, 0);
	mean.at<double>(2, 0);
}
//***********************************************************************
void imagedemo::draw_rectangle_demo(Mat &image)
{
	Rect rect;
	rect.x = 100;
	rect.y = 100;
	rect.width = 250;
	rect.height = 300;

	Mat background = Mat::zeros(image.size(),image.type());
	rectangle(background, rect, Scalar(0, 0, 255), -1,8, 0);//����������
	circle(background, Point(350, 400), 20, Scalar(255, 0, 0), -1, 8, 0);//�������Բ��
	line(background,Point(100,100),Point(350,400),Scalar(0,255,0),4,LINE_AA,0);//���ƿ����ֱ��
	RotatedRect rrt;
	rrt.center = Point(200, 200);
	rrt.size = Size(100, 200);//(�����ᣬ�̰���)
	rrt.angle = 90.0;
	ellipse(background, rrt, Scalar(0, 255, 255), 2, 8);//���Ʊ߿�ʽ��Բ
	Mat dst;
	addWeighted(image,0.7,background,0.3,0,dst);//��bg 30%��͸���ȸ�����image��
	imshow("��ͼ��ʾ",background);
}
//***********************************************************************
void imagedemo::random_draw_demo()
{
	Mat canvas = Mat::zeros(Size(512, 512), CV_8UC3);//�����հ׻���
	int w = canvas.cols;
	int h = canvas.rows;
	RNG rng(time(0));//����ϵͳʱ�䴴�����������
	while (true)
	{
		int c = waitKey(100);
		if (c == 27)
			break;
		int x1 = rng.uniform(0, w);//����߿�
		int y1 = rng.uniform(0, h);
		int x2 = rng.uniform(0, w);
		int y2 = rng.uniform(0, h);
		int b = rng.uniform(0, 255);//�����ɫ
		int g = rng.uniform(0, 255);
		int r = rng.uniform(0, 255);
		line(canvas,Point(x1,y1),Point(x2,y2),Scalar(b,g,r),1,LINE_AA,0);
		imshow("�����ɫֱ�߻���",canvas);
	}

}
//***********************************************************************
void imagedemo::polyline_demo()
{
	Mat canvas = Mat::zeros(Size(512, 512), CV_8UC3);//�����հ׻���
	Point p1(100, 100);
	Point p2(350, 100);
	Point p3(450, 280);
	Point p4(320, 450);
	Point p5(100, 100);
	std::vector<Point>pts;
	pts.push_back(p1);//���
	pts.push_back(p2);
	pts.push_back(p3);
	pts.push_back(p4);
	pts.push_back(p5);
    //�����  �ڻ��Ʊ�
	fillPoly(canvas,pts,Scalar(255,0,255),8,0);
	polylines(canvas,pts,true,Scalar(0,255,255),2,LINE_AA,0);

	std::vector<std::vector<Point>>contours;  //��ά����  ÿһ��Ԫ�ش���һ������ε㼯
	contours.push_back(pts);//�洢һ������ε㼯
	drawContours(canvas,contours,0,Scalar(0.255,255),2,8,0); //��������Ϊ����ε㼯���  ��0��ʼ
	imshow("����λ���", canvas);
}
//***********************************************************************
//����ͼ���ע���  �����ǩ 
Point sp(-1, -1);//��ʼ����ʼλ��
Point ep(-1, -1);//��ʼ���յ�λ��
Mat temp;
static void on_draw(int event, int x, int y, int flags, void*userdata)
{
	Mat image = *((Mat*)userdata);//������Ĳ���תΪmat��������
	if (event == EVENT_LBUTTONDOWN)//���������
	{
		sp.x = x;
		sp.y = y;
		std::cout << "start point:" << sp << std::endl;

	}
	else if (event == EVENT_LBUTTONUP)//����ɿ����
	{
		ep.x = x;
		ep.y = y;
		int dx = ep.x - sp.y;//w
		int dy = ep.x - sp.y;//h
		if (dx > 0 && dy > 0)
		{
			Rect box(sp.x,sp.y,dx,dy);
			rectangle(image, box, Scalar(0, 0, 255),2,8, 0);//���ƾ���
			imshow("������", image);
			imshow("ROI����", image(box));
			sp.x = -1;
			sp.y = -1;
		}
	}
	else if (event == EVENT_MOUSEMOVE)//����ƶ��¼�
	{
		if (sp.x > 0 && sp.y > 0)
		{
			ep.x = x;
			ep.y = y;
			int dx = ep.x - sp.y;//w
			int dy = ep.x - sp.y;//h
			if (dx > 0 && dy > 0)
			{
				Rect box(sp.x, sp.y, dx, dy);
				rectangle(image, box, Scalar(0, 0, 255), 2, 8, 0);//���ƾ���
				imshow("������", image);
			}
		}
	}
}
void imagedemo::mouse_drawing_demo(Mat &image)
{
	namedWindow("������", WINDOW_AUTOSIZE);
	
	setMouseCallback("������", on_draw, (void*)(&image));//��������¼�  ����ص�����
	imshow("������",image);
	temp = image.clone();
}

//***********************************************************************
void imagedemo::normal_demo(Mat &image)
{
	Mat dst;
	std::cout << image.type() << std::endl;
	image.convertTo(image, CV_32F);//ת��Ϊ������  ��һ����ǰ�� �������
	std::cout << image.type() << std::endl;
	normalize(image,dst,1.0,0,NORM_MINMAX);//����minmax��һ����ʽ ��һ����0-1֮��
	std::cout << dst.type() << std::endl;
	imshow("ͼ���һ��", dst);
}
//***********************************************************************
static void on_resize(int event, int x, int y, int flags, void*userdata)
{
	Mat image = *((Mat*)userdata);//������Ĳ���תΪmat��������
	if (event == EVENT_MOUSEHWHEEL)//������ǰ����  �Ŵ�
	{
		Mat zoomout;
		int w = image.cols;
		int h = image.rows;
		resize(image,zoomout,Size(w*1.5,h*1.5),0,0,INTER_LINEAR);//���Բ�ֵ��
		imshow("��껬������", zoomout);
	}
	else if (event == EVENT_MOUSEWHEEL)//�����ֺ󻬶�  ��С
	{
		Mat zoomin;
		int w = image.cols;
		int h = image.rows;
		resize(image, zoomin, Size(w/2, h/2), 0, 0, INTER_LINEAR);//���Բ�ֵ��
		imshow("��껬������", zoomin);
	}
}

void imagedemo::resize_demo(Mat &image)
{
	namedWindow("��껬������", WINDOW_AUTOSIZE);

	setMouseCallback("��껬������", on_resize, (void*)(&image));//��������¼�  ����ص�����
	imshow("��껬������", image);
}
//***********************************************************************
void imagedemo::flip_demo(Mat &image)
{
	Mat dst;
	while (true)
	{
		int c = waitKey(100);//100msѭ���������̲����ź�
		if (c == 27)
		{
			std::cout << "�˳�" << std::endl;
			break;
		}
		if (c == 49) //����cֵ��С   ��ͨ�����̲��Եó�
		{
			std::cout << "enter key #1" << std::endl;
			flip(image, dst, 0);//���·�ת
			//imshow("���·�ת", dst);
		}
		if (c == 50)
		{
			std::cout << "enter key #2" << std::endl;
			flip(image, dst, 0);//���ҷ�ת
			//imshow("���ҷ�ת", dst);
		}
		if (c == 51)
		{
			std::cout << "enter key #3" << std::endl;
			flip(image, dst, -1);//180�ȷ�ת
			//imshow("�Խ��߷�ת", dst);
		}
		imshow("������Ӧ��ת����", dst);
	}
}
//***********************************************************************
//��ͼƬ������ת �ҳߴ籣�ֲ���
void imagedemo::rotate_demo(Mat &image)
{
	Mat dst, M;
	int w = image.cols;
	int h = image.rows;
	M = getRotationMatrix2D(Point2f(w / 2, h / 2), 45,1.0);
	double cos = abs(M.at<double>(0,0));
	double sin = abs(M.at<double>(0, 1));
	int nw = cos * w + sin * h;
	int nh = sin * w + cos * h;
	M.at<double>(0, 2) += (nw / 2 - w / 2);
	M.at<double>(0, 2) += (nh / 2 - h / 2);
	warpAffine(image,dst,M,image.size(),INTER_LINEAR,0,Scalar(0,255,255));//ִ��ӳ��ͶӰ�任  ����Ϊcalar(0,255,255)
	imshow("��ת��ʾ",dst);
}
//***********************************************************************
void imagedemo::video_demo()
{
	VideoCapture capture(0);//0 or 1Ϊ����ͷ�豸�� ���滻Ϊ��·��/��Ƶ�ļ�.mp4(.avi)�� 
	int frame_width = capture.get(CAP_PROP_FRAME_WIDTH);//��ȡ��Ƶ�ļ��Ŀ�
	int frame_height = capture.get(CAP_PROP_FRAME_HEIGHT);//��ȡ��Ƶ�ļ��ĸ�
	int frame_cout = capture.get(CAP_PROP_FRAME_COUNT);//��ȡ��Ƶ�� ��֡��
	double fps = capture.get(CAP_PROP_FPS);//��ȡ֡��  ÿ���ȡͼƬ��
	std::cout <<"width:"<<frame_width << std::endl;
	std::cout << "height:" << frame_height << std::endl;
	std::cout << "number of frame:" << frame_cout << std::endl;
	std::cout << "fps:" << fps << std::endl;

	VideoWriter writer("F:/test.mp4",capture.get(CAP_PROP_FOCUS),fps,Size(frame_width,frame_height),true);//��һ���ı����ʽ��֡�ʣ��ߴ��С����д��洢2
	Mat frame;
	while (true)
	{
		capture.read(frame);
		flip(frame,frame,1);
		if (frame.empty())
		{
			break;
		}
		imshow("frame",frame);

		int c = waitKey(1);
		if (c == 27)
			break;
	}
	capture.release();//���ý��� �ͷ���Ƶ��
	writer.release();
}

//***********************************************************************
void imagedemo::histogram_demo(Mat &image)
{
	//��ͨ������
	std::vector<Mat>bgr_plane;
	split(image, bgr_plane);
	//�����������
	const int channels[] = { 0 };//����ͨ����Ϊ1
	const int bins[] = {256};//����ֱ��ͼ�Ҷȵȼ�Ϊ256
	float hranges[] = {0,255};//���ûҶ�ֱ��ͼͼ��Χ 0-255
	const float* ranges[] = {hranges};//��һͨ�� �Ҷ�ֱ��ͼ��Χ0-255
	Mat b_hist;
	Mat g_hist;
	Mat r_hist;
	//����B,G,Rֱ��ͼ
	calcHist(&bgr_plane[0],1,0,Mat(),b_hist,1,bins,ranges);
	calcHist(&bgr_plane[1], 1, 0, Mat(), b_hist, 1, bins, ranges);
	calcHist(&bgr_plane[2], 1, 0, Mat(), b_hist, 1, bins, ranges);
	//��ʾֱ��ͼ
	int hist_w = 512;
	int hist_h = 400;
	int bin_w = cvRound((double)hist_w/bins[0]);
	Mat histImage = Mat::zeros(hist_h,hist_w,CV_8UC3);//�������� ����ֱ��ͼ

	//��һ��ֱ��ͼ����
	normalize(b_hist,b_hist,0,histImage.rows,NORM_MINMAX,-1,Mat());
	normalize(g_hist, g_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
	normalize(r_hist, r_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());

	//����ֱ��ͼ����
	//���ƹ�����ע�� ԭ�����������Ͻ�Ϊ���� ����
	for (int i = 1; i < bins[0]; i++)
	{
		line(histImage, Point(bin_w*(i - 1), hist_h - cvRound(b_hist.at<float>(i - 1))), Point(bin_w*(i), hist_h - cvRound(b_hist.at<float>(i))), Scalar(255, 0, 0), 2, 8, 0);
		line(histImage, Point(bin_w*(i - 1), hist_h - cvRound(g_hist.at<float>(i - 1))), Point(bin_w*(i), hist_h - cvRound(g_hist.at<float>(i))), Scalar(255, 0, 0), 2, 8, 0);
		line(histImage, Point(bin_w*(i - 1), hist_h - cvRound(r_hist.at<float>(i - 1))), Point(bin_w*(i), hist_h - cvRound(r_hist.at<float>(i))), Scalar(255, 0, 0), 2, 8, 0);
	}
	//cvround  ʵ�ֽ���������ת��Ϊ����
	//��ʾֱ��ͼ
	namedWindow("ֱ��ͼ",WINDOW_AUTOSIZE);
	imshow("ֱ��ͼ",histImage);
}

//***********************************************************************
void imagedemo::histogram_2d_demo(Mat &image)
{
	Mat hsv, hs_hist;
	cvtColor(image,hsv,COLOR_BGR2HSV);
	int hbins = 30, sbins = 32;
	int hist_bins[] = {hbins,sbins};
	float h_range[] = { 0,180 };//h��  ��Χ0-180
	float s_range[] = { 0,256 };//s��  ��Χ0-256
	const float* hs_ranges[] = {h_range,s_range};
	int hs_channels[] = {0,1};
	calcHist(&hsv, 1, hs_channels, Mat(), hs_hist, 2, hist_bins, hs_ranges,true,false);
	double maxval = 0;
	minMaxLoc(hs_hist,0,&maxval,0,0);
	int Scale = 10;
	Mat hist2d_image = Mat::zeros(sbins*Scale,hbins*Scale,CV_8UC3);
	for (int h = 0; h < hbins; h++)
	{
		for (int s = 0; s < sbins; s++)
		{
			float binval = hs_hist.at<float>(h, s);
			int intensity = cvRound(binval*255/maxval);
			rectangle(hist2d_image,Point(h*Scale,s*Scale),Point((h+1)*Scale-1,(s+1)*Scale-1),Scalar::all(intensity),-1);

		}

	}
	imshow("hsvֱ��ͼ",hist2d_image);
}

//***********************************************************************
//ֱ��ͼ���⻯ ������ǿͼ��Աȶ� ͼ���������
void imagedemo::histogram_eq_demo(Mat &image)
{
	Mat gray,dst;
	cvtColor(image,gray,COLOR_BGR2GRAY);
	imshow("�Ҷ�ͼ",gray);
	equalizeHist(gray, dst);
	imshow("ֱ��ͼ���⻯", dst);
}

//***********************************************************************
void imagedemo::conv_demo(Mat &image)
{
	Mat dst;
	blur(image,dst,Size(3,3),Point(-1,-1));//����˴�СΪ3*3  ��������������ĵ��滻����Ե���ı�  �ߴ���ԭͼ������ͬ
	//������� ͼ��ģ����
	imshow("ͼ��ģ��",dst);

}

//***********************************************************************
void imagedemo::gaussian_blur_demo(Mat &image)
{
	Mat dst;
	GaussianBlur(image,dst,Size(5,5),15);  //��˹  ����sigma=15  ���Ժ���size ����sigma�ɼ���size���ڴ�С sigmaԽ�� ͼ��Խģ��
	imshow("��˹ģ��",dst);

}

//***********************************************************************
void imagedemo::bifilter_demo(Mat &image)//��˹˫��ģ��   ����Ե�����˲���
{
	Mat dst;
	bilateralFilter(image,dst,0,100,10);  //����color_sigma=100 space_sigma=10  ���Ժ���size(��d=0) ����sigma�ɼ���size���ڴ�С color_sigmaԽ�� ͼ���Ե����Խ��
	imshow("��˹˫��ģ��", dst);
}

//***********************************************************************
void imagedemo::face_detection_demo() 
{
	string root_dir = "F:/������Ӿ�/OpenCv��/opencv/sources/samples/dnn/face_detector";
	dnn::Net net = dnn::readNetFromCaffe(root_dir+"solver.prototxt");//����ģ��
	VideoCapture capture(0);//����video0 Ĭ������ͷ
	Mat frame;
	while (true)
	{
		capture.read(frame);
		flip(frame, frame, 1);//�ԳƷ�ת �������ͷ�������� �����������Ӿ�
		if (frame.empty())
		{
			break;
		}
		Mat blob = dnn::blobFromImage(frame,1.0,Size(300,300),Scalar(104,177,123),false,false);//��������ͼƬԤ���� ����ͼƬ��С300*300  ��ɫ��ֵ��
		net.setInput(blob);
		Mat probs = net.forward();//ǰ�򴫲� ���Ԥ����
		Mat detectionMat(probs.size[2],probs.size[3],CV_32F,probs.ptr<float>());//�洢Ԥ����
		//probs[0] ->����ͼƬ���
		//probs[1] ->����ͼƬ����batch
		//probs[2] ->����ͼƬԤ�����
		//probs[3] ->����ͼƬԤ���λ��

		//������� ��ʾ
		for (int i = 0; i < detectionMat.rows; i++) //i ->rows ��ʾ��Ƶ���ļ������ν�������Ԥ���ÿһ֡ͼƬ
		{
			float confidence = detectionMat.at<float>(i,2);//Ԥ�����
			if (confidence > 0.5)
			{
				int x1 = static_cast<int>(detectionMat.at<float>(i, 3)*frame.cols);
				int y1 = static_cast<int>(detectionMat.at<float>(i, 4)*frame.rows);
				int x2 = static_cast<int>(detectionMat.at<float>(i, 5)*frame.cols);
				int y2 = static_cast<int>(detectionMat.at<float>(i, 6)*frame.rows);
				Rect box(x1,y1,x2-x1,y2-y1);
				rectangle(frame,box,Scalar(0,255,0),2,8);
			}
		}
		imshow("frame", frame);
		int c = waitKey(1);
		if (c == 27)
			break;
	}
	capture.release();//���ý��� �ͷ���Ƶ��
}