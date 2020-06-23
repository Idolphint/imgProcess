#include <stdio.h>
#include <iostream>
#include <math.h>
#include <string>
#include <complex>
#include <opencv2/core.hpp>           //Mat         核心库
#include <opencv2/imgcodecs.hpp>        //imread    读图片函数
#include <opencv2/highgui.hpp>            //namedWindow imshow waitKey    界面
#include <opencv2/imgproc.hpp>            //图像处理
#include<ctime>
#define PI 3.1415926
using namespace cv;
using namespace std;
const double pi_x2 = 2 * PI;

double DFT_com(Mat& img, vector<vector<complex<double>>>& out) {
	int M = img.rows;
	int N = img.cols;
	double max = 0;
	vector<complex<double>> back_Store(M);
	for (int v = 0; v < N; v++) {
		for (int x = 0; x < M; x++) {
			complex<double> back = 0;
			for (int y = 0; y < N; y++) {
				double b_index = -2 * PI * v * y / N;
				complex<double> s_idx(0, b_index);
				if ((y + x) % 2 == 0)
					back += (double)img.at<uchar>(x, y) * exp(s_idx);
				else
					back -= (double)img.at<uchar>(x, y) * exp(s_idx);
			}
			back_Store[x] = back;
		}
		//cout << v << endl;
		vector<complex<double>> buf_line;
		for (int u = 0; u < M; u++) {
			complex<double> head = 0;
			for (int x = 0; x < M; x++) {
				double h_index = -2 * PI * u * x / M;
				complex<double> f_idx(0, h_index);
				head += back_Store[x] * exp(f_idx);
			}
			if (abs(head) > max)
				max = abs(head);
			buf_line.push_back(head);
			//out.at<Point2d>(u, v) = Point2d(head);
		}
		
		out.push_back(buf_line);
	}
	cout << "finish complex" << endl;
	return max;
}

void DFT_viu(Mat& img) {
	int M = img.rows;
	int N = img.cols;
	Mat out(M, N, CV_8U);
	cout << M << N << endl;
	double p_min = 10000000;
	double p_max = -10000000;
	vector<complex<double>> back_Store(M);
	for (int v = 0; v < N; v++) {
		for (int x = 0; x < M; x++) {
			complex<double> back = 0;
			for (int y = 0; y < N; y++) {
				double b_index = -2 * PI * v * y / N;
				complex<double> s_idx(0, b_index);
				if ((y+x)%2 == 0)
					back += (double)img.at<uchar>(x, y) * exp(s_idx);
				else
					back -= (double)img.at<uchar>(x, y) * exp(s_idx);
			}
			back_Store[x] = back;
		}
		cout << v << endl;
		for (int u = 0; u < M; u++) {
			complex<double> head = 0;
			for (int x = 0; x < M; x++) {
				double h_index = -2 * PI * u * x / M;
				complex<double> f_idx(0, h_index);
				head += back_Store[x] * exp(f_idx);
			}
			//cout << abs(head)<<"        "<<head << endl;
			head *= (double)100 / (double)(N * M);
			out.at<uchar>(u, v) = (uchar)abs(head) ;
			if (abs(head) < p_min)
				p_min = abs(head);
			else if (abs(head) > p_max)
				p_max = abs(head);
		}
	}

	imwrite("E:/2020-Spring/图像处理-模式识别/img/DFT_pinpu.jpg", out);
	imshow("pinPu", out);
	waitKey();
}


void ReverseDFT2(Mat& img) {
	int M = img.rows;
	int N = img.cols;
	vector<vector<complex<double>>> pinp(M);
	for (int v = 0; v < N; v++) {
		vector<complex<double>> buf_line;
		for (int u = 0; u < M; u++) {
			complex<double> sum = 0;
			for (int y = 0; y < N; y++) {
				for (int x = 0; x < M; x++) {
					complex<double> tmp = { 0,-1 * (pi_x2 * v * y / N + pi_x2 * u * x / M) };
					if ((y + x) % 2 == 0)
						sum += (double)img.at<uchar>(x, y) * exp(tmp);
					else
						sum -= (double)img.at<uchar>(x, y) * exp(tmp);
				}
			}
			buf_line.push_back(sum);
		}
		pinp.push_back(buf_line);
	}
	Mat outImg(M, N, CV_8U);
	for (int x = 0; x < M; x++) {
		for (int y = 0; y < N; y++) {
			complex<double> pix = 0;
			for (int u = 0; u < M; u++) {
				for (int v = 0; v < N; v++) {
					complex<double> idx{ 0,  2 * PI * (u * x / M + v * y / N) };
					complex<double>pinp_i = pinp[v][u];
					pix += pinp_i * exp(idx);
				}
			}

			pix /= (M * N);
			
			outImg.at<uchar>(x, y) = (uchar)abs(pix);
		}
	}
}

void ReverseDFT(Mat& img) {
	int M = img.rows;
	int N = img.cols;
	int len_u = M/2;
	int len_v = N / 2;
	int u_b = 0;// M / 2 - len_u;
	int u_e = M;// / 2 + len_u;
	int v_b = 0;// N / 2 - len_v;
	int v_e = N;// / 2 + len_v;

	int r = M / 15*7;

	cout << M << " " << N << endl;
	Mat outImg(M, N, CV_8U);
	vector<vector<complex<double>>> pinp;
	double max = DFT_com(img, pinp);
	cout << "max abs is"<<max << endl;
	double p_max = -10000000;
	double p_min = 10000000;
	Mat buf(M, N, CV_64F);
	int cnt;
	vector<complex<double>> recd(M);
	for (int y = 0; y < N; y++) {
		cnt = 0;
		for (int u = u_b; u < u_e; u++) {
			recd[u] = 0;
			/*int v_half = sqrt(pow(r, 2) - pow(u - (M / 2), 2));
			v_b = N / 2 - v_half;
			v_e = N / 2 + v_half;*/
			for (int v = v_b; v < v_e; v++) {
				complex<double> idx{ 0,  2 * PI * v * y / N };
				//complex<double> pinp_i(pinp[u][ v]);
				//complex<double>pinp_i = 0;
				complex<double>pinp_i = pinp[v][u];
				cnt++;
				/*if (abs(pinp[v][u]) > max/400) {
					pinp_i = pinp[v][u];
					cnt++;
				}*/
				
				//cout << pinp[u][v] << endl;
				recd[u] += pinp_i * exp(idx);
			}
		}

		for (int x = 0; x < M; x++) {
			complex<double> pix = 0;
			for (int u = u_b; u < u_e; u++) {
				complex<double> idx{ 0, 2 * PI * u * x / M };
				pix += exp(idx) * recd[u];
			}
			outImg.at<uchar>(x, y) = (uchar)(abs(pix)/(M*N));

		}
	}
	cout << 1. * cnt / (M * N) << endl;
	//imwrite("E:/2020-Spring/图像处理-模式识别/img/DFT_Lvbo"+to_string(r)+"_"+to_string(1. * cnt / (M * N))+".jpg", outImg); //to_string(len_u)+to_string(len_v)
	//imshow("reverse", outImg);
	//waitKey();
}

double C(int u) {
	if (u == 0)
		return (1 / sqrt(2));
	else
		return 1;
}

Mat Cosin(Mat& img, double max_p) {
	int M = img.rows;
	int N = img.cols;
	Mat cos_out(M, N, CV_8U);
	Mat realCos(M, N, CV_64F);
	double p_max = -10000;
	double p_min = 10000;
	for (int u = 0; u < M; u++) {
		for (int v = 0; v < N; v++) {
			double pix = 0;
			for (int x = 0; x < M; x++) {
				for (int y = 0; y < N; y++) {
					pix += cos((2 * x + 1) * u * PI / (2 * M)) *
						cos((2 * y + 1) * v * PI / (2 * N)) * img.at<uchar>(x, y);
				}
			}
			
			pix *= (2 / sqrt(M * N)) *C(u) *C(v);
			if (pix > max_p)
				max_p = pix;
			//pix *= (100. / (M * N));
			
			//cout << pix << endl;
			realCos.at<double>(u, v) = pix;
			cos_out.at<uchar>(u, v) = (uchar)(100. * pix /(M*N));
			//cout << realCos.at<double>(u, v) << endl;
		}
	}
	//cout << p_max << "  " << p_min << endl;
	/*for (int u = 0; u < M; u++) {
		for (int v = 0; v < N; v++) {
			cos_out.at<double>(u, v) = (cos_out.at<double>(u, v) - p_min) / (p_max - p_min) * 255.;
		}
	}*/
	/*imwrite("E:/2020-Spring/图像处理-模式识别/img/DCT_full.jpg", cos_out);
	imshow("cosin", cos_out);
	waitKey();*/
	return realCos;
}

void ReverseDCT(Mat& Pinpu, double max_p) {
	int M = Pinpu.rows;
	int N = Pinpu.cols;
	Mat aftChange(M, N, CV_8U);
	int cnt = 0;
	cout << "begin IDCT" << endl;
	for (int x = 0; x < M; x++) {
		for (int y = 0; y < N; y++) {
			double pix = 0;
			cnt = 0;
			for (int u = 0; u < M; u++) {
				for (int v = 0; v < N; v++) {
					//double tail = cos((2 * y + 1) * v * PI / (2 * N));
					if (Pinpu.at<double>(u, v) ) {
						pix += Pinpu.at<double>(u, v) *
							cos((2 * x + 1) * u * PI / (2 * M)) *
							cos((2 * y + 1) * v * PI / (2 * N)) * C(u) * C(v);
						cnt++;
					}											
				}
			}
			
			pix *= (2. / sqrt(M * N));
			//cout << "IDCT "<<x<<", "<<y << endl;
			aftChange.at<uchar>(x, y) = (uchar)pix;
		}
	}
	imwrite("E:/2020-Spring/图像处理-模式识别/img/IDCT_"+to_string(1.*cnt/(M*N))+".jpg", aftChange);
	imshow("changeCos", aftChange);
	waitKey();
	
}

void FFT_1(Mat& img, int x) {
	int M = img.rows;
	int N = img.cols;
	double arg = -2 * PI / N;
	double treal = cos(arg);
	double timag = sin(arg);
}


void compressDCT(Mat& gray) {
	Mat Revs(gray.rows, gray.cols, CV_64F);
	double pix_m = 0;
	Revs = Cosin(gray, pix_m);
	ReverseDCT(Revs, pix_m);
}

void main() {
	const char* path = "E:/2020-Spring/图像处理-模式识别/img/lisa-lite.png";
	Mat img = imread(path);
	Mat gray;
	
	cvtColor(img, gray, COLOR_BGR2GRAY);
	//resize(gray, gray, Size(int(img.cols/4), int(img.rows / 4)));
	//imwrite("E:/2020-Spring/图像处理-模式识别/img/ori_gray.jpg", gray);
	imshow("gray", gray);
	waitKey();
	clock_t startTime, endTime;
	startTime = clock();
	compressDCT(gray);
	//ReverseDFT(gray);
	//DFT_viu(gray);
	endTime = clock();
	cout << "The run time is: " << (double)(endTime - startTime) / CLOCKS_PER_SEC << endl;
	
}