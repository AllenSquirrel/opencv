#pragma once
#include<opencv2/opencv.hpp>
#include<opencv2/highgui.hpp>
using namespace cv;


class imagedemo {
public:
	void colorspace_demo(Mat &image);//ͼ��ɫ�ʿռ�ת��
	void mat_create_demo(Mat &image);//ͼ�����Ĵ����븳ֵ
	void pixel_visit_demo(Mat &image);//����ͼ�����ز���
	void pixel_op_demo(Mat &image);//ͼ��������������
	void tracking_bar_demo(Mat &image);//���������ڲ���
	void key_demo(Mat &image);//������Ӧ����
	void color_style_demo(Mat &image);//ͼƬ���ת��
	void bitwise_demo(Mat &image);//ͼ�� �߼�����
	void channels_demo(Mat &image);//ͼ��ͨ������
	void inrange_demo(Mat &image);//t��ȡָ��ɫ�ʷ�Χ����
	void pixel_statistic_demo(Mat &image);//����ͼ�����ز���
	void draw_rectangle_demo(Mat &image);//���λ��Ʋ���
	void random_draw_demo();//�������ͼ�μ���ɫ
	void polyline_demo();//����λ��Ƽ����
	void mouse_drawing_demo(Mat &image);//����ͼ��Ӧ����
	void normal_demo(Mat &image);//ͼ�����ع�һ��
	void resize_demo(Mat &image);//ͼ��Ŵ�����С
	void flip_demo(Mat &image);//ͼ��ת
	void rotate_demo(Mat &image);//ͼ����ת
	void video_demo();//����ͷor��Ƶ�ļ�����
	void histogram_demo(Mat &image);//����ֱ��ͼ
	void histogram_2d_demo(Mat &image);//���ƶ�άֱ��ͼ
	void histogram_eq_demo(Mat &image);//ֱ��ͼ���⻯
	void conv_demo(Mat &image);//�������
	void gaussian_blur_demo(Mat &image);//��˹ģ��
	void bifilter_demo(Mat &image);//��˹˫��ģ��   ����Ե�����˲���
	void face_detection_demo();//�������
};