#pragma once
#include<opencv2/opencv.hpp>
#include<opencv2/highgui.hpp>
using namespace cv;


class imagedemo {
public:
	void colorspace_demo(Mat &image);//图像色彩空间转换
	void mat_create_demo(Mat &image);//图像对象的创建与赋值
	void pixel_visit_demo(Mat &image);//遍历图像像素操作
	void pixel_op_demo(Mat &image);//图像像素算术操作
	void tracking_bar_demo(Mat &image);//滚动条调节操作
	void key_demo(Mat &image);//键盘响应操作
	void color_style_demo(Mat &image);//图片风格转换
	void bitwise_demo(Mat &image);//图像 逻辑操作
	void channels_demo(Mat &image);//图像通道操作
	void inrange_demo(Mat &image);//t提取指定色彩范围区域
	void pixel_statistic_demo(Mat &image);//遍历图像像素操作
	void draw_rectangle_demo(Mat &image);//矩形绘制操作
	void random_draw_demo();//随机绘制图形及颜色
	void polyline_demo();//多边形绘制及填充
	void mouse_drawing_demo(Mat &image);//鼠标绘图响应操作
	void normal_demo(Mat &image);//图像像素归一化
	void resize_demo(Mat &image);//图像放大与缩小
	void flip_demo(Mat &image);//图像翻转
	void rotate_demo(Mat &image);//图像旋转
	void video_demo();//摄像头or视频文件调用
	void histogram_demo(Mat &image);//绘制直方图
	void histogram_2d_demo(Mat &image);//绘制二维直方图
	void histogram_eq_demo(Mat &image);//直方图均衡化
	void conv_demo(Mat &image);//卷积操作
	void gaussian_blur_demo(Mat &image);//高斯模糊
	void bifilter_demo(Mat &image);//高斯双边模糊   （边缘保留滤波）
	void face_detection_demo();//人脸检测
};