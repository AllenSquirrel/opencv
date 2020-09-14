#include<demo.h>
#include<opencv2/dnn.hpp> //人脸识别demo需要使用OpenCV官方提供的网络模型

using namespace std;
using namespace cv;

void imagedemo::colorspace_demo(Mat &image)
{
	Mat gray, hsv;
	cvtColor(image,hsv,COLOR_BGR2HSV);//RGB转HSV
	cvtColor(image, gray, COLOR_BGR2GRAY);//RGB转灰度
	imshow("HSV",hsv);
	imshow("GRAY", gray);
	imwrite("F:/计算机视觉/visual c++&&opencv/source/timg_hsv.png",hsv);
}
//***********************************************************************
void imagedemo::mat_create_demo(Mat &image)
{
	Mat m1, m2,m3,m4;
	m1 = image.clone();//m1与image不为同一个
	image.copyTo(m2);//m2与image不为同一个
	m3 = image;//m3与image为同一个

	m4 = Mat::zeros(image.size(), image.type());  //通过zeros矩阵形式构造图片对象  与image保持相同大小 和通道  或形式为：（size（8,8），CV_8UC3）;
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
				int pix_value = image.at<uchar>(col, row);//遍历获取操作
				image.at<uchar>(col, row) = 255 - pix_value;//像素赋值操作
			}
			if (channel == 3)
			{
				Vec3b bgr_pix_value = image.at<Vec3b>(col, row);//遍历获取操作
				image.at<Vec3b>(col, row) = 255 - bgr_pix_value[0];//像素赋值操作
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
	add(image, m, dst);//图片整体增大亮度
	imshow("加法操作", dst);

	subtract(image, m, dst);//图片整体变暗
	imshow("减法操作", dst);

	multiply(image, m, dst);//图片整体增大对比度 m.scalar>1
	imshow("乘法操作", dst);

	divide(image, m, dst);//图片整体减小对比度
	imshow("除法操作", dst);
}
//***********************************************************************
static void on_lightness(int b, void *userdata)
{
	Mat image = *((Mat*)userdata);//将传入的参数转为mat类型数据
	Mat dst = Mat::zeros(image.size(), image.type());
	Mat m = Mat::zeros(image.size(), image.type());
	addWeighted(image, 1, m, 0, b, dst);   //根据公式 x*a+y*b+c   -->image*1+m*0+b   亮度增强
	imshow("亮度与对比度调节",dst);
}
static void on_contrast(int b, void *userdata)
{
	Mat image = *((Mat*)userdata);//将传入的参数转为mat类型数据
	Mat dst = Mat::zeros(image.size(), image.type());
	Mat m = Mat::zeros(image.size(), image.type());
	double contrast = b / 200.0;
	addWeighted(image, contrast, m, 0.0, 0.0, dst);   //根据公式 x*a+y*b+c   -->image*1+m*0+b   亮度增强
	imshow("亮度与对比度调节", dst);
}
void imagedemo::tracking_bar_demo(Mat &image)
{
	namedWindow("亮度与对比度调节",WINDOW_AUTOSIZE);
	int lightness = 50;
	int maxvalue_light = 100;
	int contrast = 100;
	int maxvalue_contrast = 200;
	createTrackbar("light_bar:","亮度与对比度调节",&lightness,maxvalue_light,on_lightness,(void*)(&image));
	createTrackbar("contrast_bar:", "亮度与对比度调节", &contrast, maxvalue_contrast, on_contrast, (void*)(&image));
	on_lightness(50, &image);
	// >50 图片亮度增大   <50 图片亮度减小
	// >100 图片对比度增大   <100 图片对比度减小
}
//***********************************************************************
void imagedemo::key_demo(Mat &image)
{
	Mat dst;
	while (true)
	{
		int c = waitKey(100);//100ms循环监听键盘操作信号
		if (c == 27) 
		{
			std::cout << "退出" << std::endl;
			break;
		}
		if (c == 49) //具体c值大小   可通过键盘测试得出
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
		imshow("键盘响应操作", dst);
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
		int c = waitKey(2000);//2s循环监听
		if (c == 27)
		{
			break;
		}
		applyColorMap(image,dst,colormap[index%13]);
		index++;
		imshow("图片风格转换", dst);
	}
}
//***********************************************************************
void imagedemo::bitwise_demo(Mat &image)
{
	Mat m1 = Mat::zeros(Size(256,256),CV_8UC3);
	Mat m2 = Mat::zeros(Size(256, 256), CV_8UC3);
	rectangle(m1,Rect(100,100,80,80),Scalar(255,255,0),-1,LINE_8,0);//其中-1表示 填充式绘制矩形  1表示边框式绘制矩形
	rectangle(m1, Rect(150, 150, 80, 80), Scalar(0, 255, 255), -1, LINE_8, 0);

	imshow("m1", m1);
	imshow("m2", m2);

	Mat dst1,dst2,dst3,dst4;
	bitwise_and(m1, m2,dst1);
	imshow("与操作",dst1);
	bitwise_or(m1, m2, dst2);
	imshow("或操作", dst2);
	bitwise_xor(m1, m2, dst3);
	imshow("异或操作", dst3);
}
//***********************************************************************
void imagedemo::channels_demo(Mat &image)
{
	std::vector<Mat> mv;
	split(image, mv);//RGB通道分离
	imshow("blue",mv[0]);//B
	imshow("gree", mv[1]);//G
	imshow("red", mv[2]);//R

	Mat dst;
	mv[1] = 0;
	mv[2] = 0;//消除gree和red颜色,只保留蓝色
	merge(mv, dst);//三通道合并
	imshow("蓝色",dst);

	int from_to[] = {0,2,1,1,2,0};//保持1通道不变  0,2通道交换
	mixChannels(&image,1,&dst,1,from_to,3); //通道混合
	imshow("B-R通道混合", dst);
}
//***********************************************************************
void imagedemo::inrange_demo(Mat &image)
{
	Mat hsv, mask;
	cvtColor(image, hsv, COLOR_BGR2HSV);
	inRange(hsv,Scalar(35,43,46),Scalar(77,255,255),mask);//将不同颜色分开从绿色背景分离  此时前景图像为0（黑）背景为1或255（白）
	imshow("mask",mask);

	Mat redback = Mat::zeros(image.size(), image.type());
	redback = Scalar(40,40,200);//红色背景
	bitwise_not(mask, mask); //取反操作  此时前景图像为白  背景为0（黑）
	imshow("roi区域提取",redback);
}

//***********************************************************************
void imagedemo::pixel_statistic_demo(Mat &image)
{
	double minv, maxv;
	Point minloc, maxloc;
	std::vector<Mat>mv;
	split(image,mv);//图像按通道进行拆分  此时RGB 三通道拆分开
	for(int i=0; i < mv.size(); i++)
	{
		minMaxLoc(mv[i],&minv,&maxv,&minloc,&maxloc,Mat());//求最大 最小像素值
		std::cout << "No.channels:" << i << "min_value:" << minv << "max_value:" << maxv << std::endl;
	}
	Mat mean, stddev;
	meanStdDev(image, mean, stddev);//求像素均值 方差  每一个通道对于一个均值和方差
	std::cout << "mean:" << mean << "stddev:" << stddev << std::endl;
	mean.at<double>(0, 0);//取3*1维中第一个均值  即第一个通道的像素均值
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
	rectangle(background, rect, Scalar(0, 0, 255), -1,8, 0);//绘制填充矩形
	circle(background, Point(350, 400), 20, Scalar(255, 0, 0), -1, 8, 0);//绘制填充圆形
	line(background,Point(100,100),Point(350,400),Scalar(0,255,0),4,LINE_AA,0);//绘制抗锯齿直线
	RotatedRect rrt;
	rrt.center = Point(200, 200);
	rrt.size = Size(100, 200);//(长半轴，短半轴)
	rrt.angle = 90.0;
	ellipse(background, rrt, Scalar(0, 255, 255), 2, 8);//绘制边框式椭圆
	Mat dst;
	addWeighted(image,0.7,background,0.3,0,dst);//以bg 30%的透明度附着在image上
	imshow("绘图显示",background);
}
//***********************************************************************
void imagedemo::random_draw_demo()
{
	Mat canvas = Mat::zeros(Size(512, 512), CV_8UC3);//创建空白画布
	int w = canvas.cols;
	int h = canvas.rows;
	RNG rng(time(0));//根据系统时间创建随机数种子
	while (true)
	{
		int c = waitKey(100);
		if (c == 27)
			break;
		int x1 = rng.uniform(0, w);//随机高宽
		int y1 = rng.uniform(0, h);
		int x2 = rng.uniform(0, w);
		int y2 = rng.uniform(0, h);
		int b = rng.uniform(0, 255);//随机颜色
		int g = rng.uniform(0, 255);
		int r = rng.uniform(0, 255);
		line(canvas,Point(x1,y1),Point(x2,y2),Scalar(b,g,r),1,LINE_AA,0);
		imshow("随机颜色直线绘制",canvas);
	}

}
//***********************************************************************
void imagedemo::polyline_demo()
{
	Mat canvas = Mat::zeros(Size(512, 512), CV_8UC3);//创建空白画布
	Point p1(100, 100);
	Point p2(350, 100);
	Point p3(450, 280);
	Point p4(320, 450);
	Point p5(100, 100);
	std::vector<Point>pts;
	pts.push_back(p1);//入队
	pts.push_back(p2);
	pts.push_back(p3);
	pts.push_back(p4);
	pts.push_back(p5);
    //先填充  在绘制边
	fillPoly(canvas,pts,Scalar(255,0,255),8,0);
	polylines(canvas,pts,true,Scalar(0,255,255),2,LINE_AA,0);

	std::vector<std::vector<Point>>contours;  //二维容器  每一个元素代表一个多边形点集
	contours.push_back(pts);//存储一个多边形点集
	drawContours(canvas,contours,0,Scalar(0.255,255),2,8,0); //第三参数为多边形点集序号  从0开始
	imshow("多边形绘制", canvas);
}
//***********************************************************************
//制作图像标注软件  加入标签 
Point sp(-1, -1);//初始化起始位置
Point ep(-1, -1);//初始化终点位置
Mat temp;
static void on_draw(int event, int x, int y, int flags, void*userdata)
{
	Mat image = *((Mat*)userdata);//将传入的参数转为mat类型数据
	if (event == EVENT_LBUTTONDOWN)//鼠标左键点击
	{
		sp.x = x;
		sp.y = y;
		std::cout << "start point:" << sp << std::endl;

	}
	else if (event == EVENT_LBUTTONUP)//鼠标松开左键
	{
		ep.x = x;
		ep.y = y;
		int dx = ep.x - sp.y;//w
		int dy = ep.x - sp.y;//h
		if (dx > 0 && dy > 0)
		{
			Rect box(sp.x,sp.y,dx,dy);
			rectangle(image, box, Scalar(0, 0, 255),2,8, 0);//绘制矩形
			imshow("鼠标绘制", image);
			imshow("ROI区域", image(box));
			sp.x = -1;
			sp.y = -1;
		}
	}
	else if (event == EVENT_MOUSEMOVE)//鼠标移动事件
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
				rectangle(image, box, Scalar(0, 0, 255), 2, 8, 0);//绘制矩形
				imshow("鼠标绘制", image);
			}
		}
	}
}
void imagedemo::mouse_drawing_demo(Mat &image)
{
	namedWindow("鼠标绘制", WINDOW_AUTOSIZE);
	
	setMouseCallback("鼠标绘制", on_draw, (void*)(&image));//捕获鼠标事件  请求回调函数
	imshow("鼠标绘制",image);
	temp = image.clone();
}

//***********************************************************************
void imagedemo::normal_demo(Mat &image)
{
	Mat dst;
	std::cout << image.type() << std::endl;
	image.convertTo(image, CV_32F);//转化为浮点数  归一化的前提 必须完成
	std::cout << image.type() << std::endl;
	normalize(image,dst,1.0,0,NORM_MINMAX);//采用minmax归一化方式 归一化到0-1之间
	std::cout << dst.type() << std::endl;
	imshow("图像归一化", dst);
}
//***********************************************************************
static void on_resize(int event, int x, int y, int flags, void*userdata)
{
	Mat image = *((Mat*)userdata);//将传入的参数转为mat类型数据
	if (event == EVENT_MOUSEHWHEEL)//鼠标滚轮前滑动  放大
	{
		Mat zoomout;
		int w = image.cols;
		int h = image.rows;
		resize(image,zoomout,Size(w*1.5,h*1.5),0,0,INTER_LINEAR);//线性插值法
		imshow("鼠标滑动放缩", zoomout);
	}
	else if (event == EVENT_MOUSEWHEEL)//鼠标滚轮后滑动  缩小
	{
		Mat zoomin;
		int w = image.cols;
		int h = image.rows;
		resize(image, zoomin, Size(w/2, h/2), 0, 0, INTER_LINEAR);//线性插值法
		imshow("鼠标滑动放缩", zoomin);
	}
}

void imagedemo::resize_demo(Mat &image)
{
	namedWindow("鼠标滑动放缩", WINDOW_AUTOSIZE);

	setMouseCallback("鼠标滑动放缩", on_resize, (void*)(&image));//捕获鼠标事件  请求回调函数
	imshow("鼠标滑动放缩", image);
}
//***********************************************************************
void imagedemo::flip_demo(Mat &image)
{
	Mat dst;
	while (true)
	{
		int c = waitKey(100);//100ms循环监听键盘操作信号
		if (c == 27)
		{
			std::cout << "退出" << std::endl;
			break;
		}
		if (c == 49) //具体c值大小   可通过键盘测试得出
		{
			std::cout << "enter key #1" << std::endl;
			flip(image, dst, 0);//上下翻转
			//imshow("上下翻转", dst);
		}
		if (c == 50)
		{
			std::cout << "enter key #2" << std::endl;
			flip(image, dst, 0);//左右翻转
			//imshow("左右翻转", dst);
		}
		if (c == 51)
		{
			std::cout << "enter key #3" << std::endl;
			flip(image, dst, -1);//180度翻转
			//imshow("对角线翻转", dst);
		}
		imshow("键盘响应翻转操作", dst);
	}
}
//***********************************************************************
//绕图片中心旋转 且尺寸保持不变
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
	warpAffine(image,dst,M,image.size(),INTER_LINEAR,0,Scalar(0,255,255));//执行映射投影变换  背景为calar(0,255,255)
	imshow("旋转演示",dst);
}
//***********************************************************************
void imagedemo::video_demo()
{
	VideoCapture capture(0);//0 or 1为摄像头设备号 可替换为“路径/视频文件.mp4(.avi)” 
	int frame_width = capture.get(CAP_PROP_FRAME_WIDTH);//获取视频文件的宽
	int frame_height = capture.get(CAP_PROP_FRAME_HEIGHT);//获取视频文件的高
	int frame_cout = capture.get(CAP_PROP_FRAME_COUNT);//获取视频流 总帧数
	double fps = capture.get(CAP_PROP_FPS);//获取帧率  每秒获取图片数
	std::cout <<"width:"<<frame_width << std::endl;
	std::cout << "height:" << frame_height << std::endl;
	std::cout << "number of frame:" << frame_cout << std::endl;
	std::cout << "fps:" << fps << std::endl;

	VideoWriter writer("F:/test.mp4",capture.get(CAP_PROP_FOCUS),fps,Size(frame_width,frame_height),true);//以一定的编码格式，帧率，尺寸大小进行写入存储2
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
	capture.release();//调用结束 释放视频流
	writer.release();
}

//***********************************************************************
void imagedemo::histogram_demo(Mat &image)
{
	//三通道分离
	std::vector<Mat>bgr_plane;
	split(image, bgr_plane);
	//定义参数变量
	const int channels[] = { 0 };//设置通道数为1
	const int bins[] = {256};//设置直方图灰度等级为256
	float hranges[] = {0,255};//设置灰度直方图图范围 0-255
	const float* ranges[] = {hranges};//第一通道 灰度直方图范围0-255
	Mat b_hist;
	Mat g_hist;
	Mat r_hist;
	//计算B,G,R直方图
	calcHist(&bgr_plane[0],1,0,Mat(),b_hist,1,bins,ranges);
	calcHist(&bgr_plane[1], 1, 0, Mat(), b_hist, 1, bins, ranges);
	calcHist(&bgr_plane[2], 1, 0, Mat(), b_hist, 1, bins, ranges);
	//显示直方图
	int hist_w = 512;
	int hist_h = 400;
	int bin_w = cvRound((double)hist_w/bins[0]);
	Mat histImage = Mat::zeros(hist_h,hist_w,CV_8UC3);//创建画布 绘制直方图

	//归一化直方图数据
	normalize(b_hist,b_hist,0,histImage.rows,NORM_MINMAX,-1,Mat());
	normalize(g_hist, g_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
	normalize(r_hist, r_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());

	//绘制直方图曲线
	//绘制过程中注意 原点坐标以左上角为基础 绘制
	for (int i = 1; i < bins[0]; i++)
	{
		line(histImage, Point(bin_w*(i - 1), hist_h - cvRound(b_hist.at<float>(i - 1))), Point(bin_w*(i), hist_h - cvRound(b_hist.at<float>(i))), Scalar(255, 0, 0), 2, 8, 0);
		line(histImage, Point(bin_w*(i - 1), hist_h - cvRound(g_hist.at<float>(i - 1))), Point(bin_w*(i), hist_h - cvRound(g_hist.at<float>(i))), Scalar(255, 0, 0), 2, 8, 0);
		line(histImage, Point(bin_w*(i - 1), hist_h - cvRound(r_hist.at<float>(i - 1))), Point(bin_w*(i), hist_h - cvRound(r_hist.at<float>(i))), Scalar(255, 0, 0), 2, 8, 0);
	}
	//cvround  实现将浮点数据转化为整型
	//显示直方图
	namedWindow("直方图",WINDOW_AUTOSIZE);
	imshow("直方图",histImage);
}

//***********************************************************************
void imagedemo::histogram_2d_demo(Mat &image)
{
	Mat hsv, hs_hist;
	cvtColor(image,hsv,COLOR_BGR2HSV);
	int hbins = 30, sbins = 32;
	int hist_bins[] = {hbins,sbins};
	float h_range[] = { 0,180 };//h域  范围0-180
	float s_range[] = { 0,256 };//s域  范围0-256
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
	imshow("hsv直方图",hist2d_image);
}

//***********************************************************************
//直方图均衡化 可以增强图像对比度 图像更加清晰
void imagedemo::histogram_eq_demo(Mat &image)
{
	Mat gray,dst;
	cvtColor(image,gray,COLOR_BGR2GRAY);
	imshow("灰度图",gray);
	equalizeHist(gray, dst);
	imshow("直方图均衡化", dst);
}

//***********************************************************************
void imagedemo::conv_demo(Mat &image)
{
	Mat dst;
	blur(image,dst,Size(3,3),Point(-1,-1));//卷积核大小为3*3  卷积输出结果以中心点替换，边缘不改变  尺寸与原图保持相同
	//卷积操作 图像模糊化
	imshow("图像模糊",dst);

}

//***********************************************************************
void imagedemo::gaussian_blur_demo(Mat &image)
{
	Mat dst;
	GaussianBlur(image,dst,Size(5,5),15);  //高斯  参数sigma=15  可以忽视size 根据sigma可计算size窗口大小 sigma越大 图像越模糊
	imshow("高斯模糊",dst);

}

//***********************************************************************
void imagedemo::bifilter_demo(Mat &image)//高斯双边模糊   （边缘保留滤波）
{
	Mat dst;
	bilateralFilter(image,dst,0,100,10);  //参数color_sigma=100 space_sigma=10  可以忽视size(即d=0) 根据sigma可计算size窗口大小 color_sigma越大 图像边缘差异越大
	imshow("高斯双边模糊", dst);
}

//***********************************************************************
void imagedemo::face_detection_demo() 
{
	string root_dir = "F:/计算机视觉/OpenCv库/opencv/sources/samples/dnn/face_detector";
	dnn::Net net = dnn::readNetFromCaffe(root_dir+"solver.prototxt");//加载模型
	VideoCapture capture(0);//调用video0 默认摄像头
	Mat frame;
	while (true)
	{
		capture.read(frame);
		flip(frame, frame, 1);//对称翻转 解决摄像头镜像问题 更符合人眼视觉
		if (frame.empty())
		{
			break;
		}
		Mat blob = dnn::blobFromImage(frame,1.0,Size(300,300),Scalar(104,177,123),false,false);//输入网络图片预处理 输入图片大小300*300  颜色均值化
		net.setInput(blob);
		Mat probs = net.forward();//前向传播 输出预测结果
		Mat detectionMat(probs.size[2],probs.size[3],CV_32F,probs.ptr<float>());//存储预测结果
		//probs[0] ->输入图片序号
		//probs[1] ->输入图片所属batch
		//probs[2] ->输入图片预测概率
		//probs[3] ->输入图片预测框位置

		//解析结果 显示
		for (int i = 0; i < detectionMat.rows; i++) //i ->rows 表示视频流文件里依次进入网络预测的每一帧图片
		{
			float confidence = detectionMat.at<float>(i,2);//预测概率
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
	capture.release();//调用结束 释放视频流
}