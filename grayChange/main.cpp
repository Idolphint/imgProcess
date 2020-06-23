#include <stdio.h>
#include <string>
#include<stdlib.h>
#include<Windows.h>
#include <malloc.h>
#include <iostream>
#include <vector>
#include <math.h>

#include <opencv2/core.hpp>           //Mat         核心库
#include <opencv2/imgcodecs.hpp>        //imread    读图片函数
#include <opencv2/highgui.hpp>            //namedWindow imshow waitKey    界面
#include <opencv2\imgproc.hpp>            //图像处理

using namespace std;
using namespace cv;
const int W = 512;
const int H = 512;
//void load_img_by_c(const char* path, int*& img) {
//	/**************  废掉了 ************/
//	FILE* fn;
//	fn = fopen(path, "r");
//	fseek(fn, 0, SEEK_END);//定位文件指针到文件尾。
//	int size = ftell(fn);
//	cout << "img size: " << size << endl;
//	rewind(fn); //返回文件头
//	//fread(img, sizeof(int), W*H, fn);
//	
//}
//
//void load_img(Mat & img, int*& res) {
//	
//	int cols = img.cols;
//	int rows = img.rows;
//	cout << img.channels() << endl;
//	for (int i = 0; i < cols; i++) {
//		for (int j = 0; j < rows; j++) {
//			res[i * rows + j] = int(img.at<char>(i, j));
//		}
//	}
//	
//	cout << "finish copy" << endl;
//}
//
//void find_max_min(int*& img, int& max, int& min) {
//	max = -1000001;
//	min = 1000001;
//	for (int i = 0; i < W * H; i++) {
//		if (img[i] > max) {
//			max = img[i];
//		}
//		else if (img[i] < min) {
//			min = img[i];
//		}
//	}
//}
//
//void average_gram(int step, int(&ori_gram)[256], vector<int>& inserts) {
//	vector<int> buf_gram;
//	for (int i = 0; i < step; i++) {
//		int num = 0;
//		int range = 256 / step;
//		if (i == step - 1)
//			range += (256 % step);
//		for (int j = 0; j < range; j++) {
//			num += ori_gram[i * (256 / step) + j];
//		}
//		buf_gram.push_back(num);
//	}
//
//	int aver_num = 512 * 512 / step;
//	
//	for (int i = 0; i < step; i++) {
//		int now_num = buf_gram.at(i);
//		while (now_num + buf_gram.at(i + 1) <= aver_num) {
//			now_num += buf_gram.at(i + 1);
//			i++;
//		}
//		inserts.push_back(now_num);
//	}
//	for (int num : inserts) {
//		cout << "\n" << num << "  " << endl;
//	}
//
//}
//
//int convert_pixel(int ori, int(&sum_table)[256], int step) {
//	return 0;
//}
//
//void show_balance_gram(int*& ori_img, vector<int> inserts, int*& dst_img) {
//	int sum_tabel[256];
//	int sum = 0;
//	int groups = inserts.size();
//	for (int i = 0; i < groups; i++) {
//		sum += inserts.at(i);
//		sum_tabel[i] = sum;
//	}
//
//	for (int i = 0; i < W * H; i++) {
//		ori_img[i] = convert_pixel(ori_img[i], sum_tabel, groups);
//	}
//}

void draw_line(Mat& img, int min_be, int max_be, int min_af, int max_af) {
	Mat liner_img = img.clone();
	for (int i = 0; i < W; i++) {
		uchar* p = liner_img.ptr<uchar>(i);
		for (int j = 0; j < H; j++) {
			if (img.at<uchar>(i, j) <= min_be)
				p[j] = (uchar)(1. * min_af * img.at<uchar>(i, j) / min_be);
			else if (img.at<uchar>(i, j) >= max_be)
				p[j] = (uchar)(1. * (255 - max_af) * (img.at<uchar>(i, j) - max_be) / (255 - max_be) + max_af);
			else
				p[j] = (uchar)(1. * (max_af - min_af) * (img.at<uchar>(i, j) - min_be) / (max_be - min_be) + min_af);

		}
	}
	namedWindow("line_img");
	imshow("line_img", liner_img);
	waitKey(0);
	imwrite("E:/2020-Spring/图像处理-模式识别/Project1_grayChange/line_img.jpg", liner_img);
}

void draw_balance(Mat& img) {
	int histogram[256] = { 0 };
	double gray_prob[256] = { 0 };
	double gray_sum[256] = { 0 };

	for (int i = 0; i < W; i++) {
		for (int j = 0; j < H; j++) {
			if (img.at<uchar>(i, j) < 0 || img.at<uchar>(i, j) > 255)
				cout << img.at<uchar>(i, j) << " overflower" << endl;
			else histogram[img.at<uchar>(i, j)] ++;
		}
	}
	double prob = 0;
	double sum_prob = 0;
	for (int i = 0; i < 256; i++) {
		gray_prob[i] = prob = 1.0 * histogram[i] / (W * H);
		sum_prob += prob;
		gray_sum[i] = sum_prob;
		//cout << gray_sum[i] << endl;
	}
	Mat balance = img.clone();
	for (int i = 0; i < W; i++) {
		uchar* p = balance.ptr<uchar>(i);
		for (int j = 0; j < H; j++) {
			uchar after_gray = gray_sum[img.at<uchar>(i, j)] * 255 + 0.5; //四舍五入
			// cout << int(after_gray) << "  ";
			p[j] = after_gray;
		}
	}
	namedWindow("after_balance");
	imshow("after_balance", balance);
	waitKey();
	imwrite("E:/2020春季-大三下/图像处理-模式识别/Project1_grayChange/after_balance.jpg", balance);
}


void draw_exp(Mat& img, float lamba) {
	Mat exp_img = img.clone();
	for (int i = 0; i < W; i++) {
		uchar* p = exp_img.ptr<uchar>(i);
		for (int j = 0; j < H; j++) {
			uchar pixel = (uchar)(pow(img.at<uchar>(i, j), lamba));
			p[j] = pixel;
		}
	}
	namedWindow("exp_img");
	imshow("exp_img", exp_img);
	waitKey(0);
	imwrite("E:/2020春季-大三下/图像处理-模式识别/Project1_grayChange/exp"+to_string(lamba)+"_img.jpg", exp_img);
}

void main() {
	const char* path = "E:/2020-Spring/图像处理-模式识别/Project1_grayChange/test1.jpg";
	int* a;
	a = (int*)malloc(sizeof(int) * 512 * 512);
	Mat img = imread(path, 0);
	//load_img(img, a);
	namedWindow("before");
	imshow("before", img);
	waitKey(0);
	
	Scalar mean;
	Scalar stddev;
	meanStdDev(img, mean, stddev);
	int mean_pxl = mean.val[0];
	int stddev_pxl = stddev.val[0];
	cout << mean_pxl <<" "<< stddev_pxl << endl;

	draw_line(img, (mean_pxl - stddev_pxl), (mean_pxl + stddev_pxl) ,  10, 245);
	//draw_exp(img, 2);

	//接下来要对直方图预设一个分级，然后分级合并，重新分配灰度值
	//for (int i = 0; i < 256; i++) cout << histogram[i] << endl;
}