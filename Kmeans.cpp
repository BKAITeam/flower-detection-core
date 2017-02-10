#include <algorithm>    // std::sort
#include <vector>       // std::vector
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/contrib/contrib.hpp>


using namespace std;
using namespace cv;

double func(const Mat& img, int j,const Point& m){
	return sqrt((int)(img.at<uchar>(j, 0) - m.x)*(int)(img.at<uchar>(j, 0) - m.x) + (int)(img.at<uchar>(j, 1) - m.y)*(int)(img.at<uchar>(j, 1) - m.y));
}

double dist(const Point& lhs, const Point& rhs)
{
	return sqrt((int)(lhs.x - rhs.x)*(int)(lhs.x - rhs.x) + (int)(lhs.y - rhs.y)*(int)(lhs.y - rhs.y));
}

//moment
double mo_ment(const Mat& img,int p, int q){
	double m = 0;
	for (int x = 0; x < img.rows; x++)
		for (int y = 0; y < img.cols; y++){
			if (img.at<uchar>(x, y) == 1){
				m += pow(x, p)*pow(y, q);
			}
		}
	return m;
}

//central moments
double cent(const Mat& img, double cenx, double ceny,int p, int q)
{
	double mu = 0;
	for (int x = 0; x < img.rows; x++)
		for (int y = 0; y < img.cols; y++){
			if (img.at<uchar>(x, y) == 1){
				mu += pow(x - cenx, p)*pow(y - ceny, q);
			}
		}
	return mu;
}

//normalized central moments
double norm_cent(double mu, double mu00, int p, int q)
{
	return mu / (pow(mu00, (float)(p + q) / 2.0 + 1));
}


int main(){
	//declare image
	Mat he = imread("C:/Users/DUC13T3/Desktop/XLA/sun/result/1sun003.jpg",CV_LOAD_IMAGE_COLOR);
	imshow("Source",he);
		
	//Change rgb to Lab
	Mat Lab_he;
	cvtColor(he, Lab_he, CV_BGR2Lab);

	//Split 3 channel L*a*b*. 
	vector<Mat> Channel;
	Mat channel[3], img;
	split(Lab_he, channel);
	//channel[0] = Mat::zeros(Size(Lab_he.rows, Lab_he.cols), CV_8UC1);
	//Channel.push_back(channel[0]);
	Channel.push_back(channel[1]);	//a* Channel
	Channel.push_back(channel[2]);	//b* Channel

	//Merge into 1 image 2 channel.
	merge(Channel, img);	
	//img.convertTo(img, CV_32FC2);

	int nrows, ncols;
	nrows = img.rows;
	ncols = img.cols;
	
	//Reshape into 1 channel
	img = img.reshape(1, nrows*ncols);	//1 Channel, 2 cols

	//Kmeans clustering element
	Point m1, m2, m3;
	Point old_m1, old_m2, old_m3;
	vector<Point> K1;
	vector<Point> K2;
	vector<Point> K3;
	vector<uchar> cluster_idx;
	double Min, x1, x2, x3;
	int rnd1, rnd2, rnd3;

	
	old_m1 = old_m2 = old_m3 = Point(img.at<uchar>(0, 0), img.at<uchar>(0, 1));

	do{
		rnd1 = rand() % img.rows;
		m1.x = img.at<uchar>(rnd1, 0);
		m1.y = img.at<uchar>(rnd1, 1);
		rnd2 = rand() % img.rows;
		m2.x = img.at<uchar>(rnd2, 0);
		m2.y = img.at<uchar>(rnd2, 1);
		rnd3 = rand() % img.rows;
		m3.x = img.at<uchar>(rnd3, 0);
		m3.y = img.at<uchar>(rnd3, 1);
	}
		while (!(m1 != m2 && m2 != m3 && m1 != m3));

	int N = 5;
	while (!(m1 == old_m1 && m2 == old_m2 && m3 == old_m3 ||N==0)){
		K1.clear();
		K2.clear();
		K3.clear();
		cluster_idx.clear();
		for (int j = 0; j < img.rows; j++){
			x1 = func(img, j, m1);
			x2 = func(img, j, m2);
			x3 = func(img, j, m3);
			Min = x1 < x2 ? x1 : x2;
			Min = Min < x3? Min : x3;
			if (Min == x1){
				cluster_idx.push_back(1);
				K1.push_back(Point(img.at<uchar>(j, 0), img.at<uchar>(j, 1)));
			}
			else if (Min == x2){
				cluster_idx.push_back(2);
				K2.push_back(Point(img.at<uchar>(j, 0), img.at<uchar>(j, 1)));
			}
			else if (Min == x3){
				cluster_idx.push_back(3);
				K3.push_back(Point(img.at<uchar>(j, 0), img.at<uchar>(j, 1)));
			}
		}
		sort(K1.begin(), K1.end(), [m1](const Point& A, const Point& B){ return dist(m1, A) < dist(m1, B); });
		sort(K2.begin(), K2.end(), [m2](const Point& A, const Point& B){ return dist(m2, A) < dist(m2, B); });
		sort(K3.begin(), K3.end(), [m3](const Point& A, const Point& B){ return dist(m3, A) < dist(m3, B); });

		old_m1 = m1;
		old_m2 = m2;
		old_m3 = m3;		
		m1 = K1.at((int)(K1.size() / 2));
		m2 = K2.at((int)(K2.size() / 2));
		m3 = K3.at((int)(K3.size() / 2));
		N--;
	}
	
	Mat A = Mat::zeros(100, 100, CV_8UC1);
	Mat B = Mat::zeros(100, 100, CV_8UC1);
	Mat C = Mat::zeros(100, 100, CV_8UC1);
	
	for (int i = 0; i < img.rows; i++){
			if (cluster_idx.at(i) == 1){
				A.at<uchar>(i / 100, i % 100) = 255;
			}
			if (cluster_idx.at(i) == 2){
				B.at<uchar>(i / 100, i % 100) = 255;
			}
			if (cluster_idx.at(i) == 3){
				C.at<uchar>(i / 100, i % 100) = 255;
			}
	}

	imshow("A", A);
	imshow("B", B);
	imshow("C", C);
	Mat dst;
	he.copyTo(dst, B);
	imshow("Mask", dst);

	/*=======================Hu's moment=========================*/
	
	//Image binary
	Mat HuImg;
	normalize(C, HuImg, 0, 1, NORM_MINMAX, CV_8UC1);

	// spatial moments
	double m[4][4];
	// central moments
	double mu[4][4];
	// central normalized moments
	double nu[4][4];
	// centroid of the image	
	double x_cen;
	double y_cen;
	// seven moment invariants Hu
	double Hu[7];

	//Calculate moment
	for (int p = 0; p <= 3; ++p){
		for (int q = 0; q <= 3 && p + q < 4; ++q){
			m[p][q] = mo_ment(HuImg, p, q);
		}
	}
	
	// centroid of the image
	x_cen = m[1][0] / m[0][0];
	y_cen = m[0][1] / m[0][0];

	//Calculate center moment
	for (int p = 0; p <= 3; ++p){
		for (int q = 0; q <= 3 && p + q < 4; ++q){
			if (p + q > 1)
				mu[p][q] = cent(HuImg, x_cen, y_cen, p, q);
		}
	}

	//The normalized central moments
	for (int p = 0; p <= 3; ++p){
		for (int q = 0; q <= 3 && p + q < 4; ++q){
			if (p + q > 1)
				nu[p][q] = norm_cent(mu[p][q], m[0][0], p, q);
		}
	}
	
	//Based on normalized central moments
	Hu[0] = nu[2][0] + nu[0][2];
	Hu[1] = (nu[2][0] - nu[0][2])*(nu[2][0] + nu[0][2]) + 4 * nu[1][1] * nu[1][1];
	Hu[2] = (nu[3][0] - 3 * nu[1][2])*(nu[3][0] - 3 * nu[1][2]) + (nu[3][0] - 3 * nu[2][1])*(nu[3][0] - 3 * nu[2][1]);
	Hu[3] = (nu[3][0] + nu[1][2])*(nu[3][0] + nu[1][2]) + (nu[2][1] + nu[0][3])*(nu[2][1] + nu[0][3]);
	Hu[4] = (nu[3][0] - 3 * nu[1][2])*(nu[3][0] + nu[1][2])*((nu[3][0] + nu[1][2])*(nu[3][0] + nu[1][2]) - 3 * (nu[2][1] + nu[0][3])*(nu[2][1] + nu[0][3]))
		+ (3 * nu[2][1] - nu[0][3])*(nu[2][1] + nu[0][3])*(3 * (nu[3][0] + nu[1][2])*(nu[3][0] + nu[1][2]) - (nu[2][1] + nu[0][3])*(nu[2][1] + nu[0][3]));
	Hu[5] = (nu[2][0] - nu[0][2])*((nu[3][0] + nu[1][2])*(nu[3][0] + nu[1][2]) - (nu[2][1] + nu[0][3])*(nu[2][1] + nu[0][3]))
		+4*nu[1][1]*(nu[3][0]+nu[1][2])*(nu[2][1]+nu[0][3]);
	Hu[6] = (3 * nu[2][1] - nu[0][3])*(nu[3][0] + nu[1][2])*((nu[3][0] + nu[1][2])*(nu[3][0] + nu[1][2]) - 3 * (nu[2][1] + nu[0][3])* (nu[2][1] + nu[0][3]))
		- (nu[3][0] - 3 * nu[1][2])*(nu[2][1] + nu[0][3])*(3 * (nu[3][0] + nu[1][2])* (nu[3][0] + nu[1][2]) - (nu[2][1] + nu[0][3])*(nu[2][1] + nu[0][3]));

	for (int i = 0; i < 7; i++)
		cout << Hu[i] << endl;


	cout << "Chuyen gia tri dac trung Hu's sang logaric"<<endl;
	for (int i = 0; i < 7; i++){
		Hu[i] = log(abs(Hu[i])) / log(exp(1));
	}
	for (int i = 0; i < 7; i++){
		cout << Hu[i] << " " << endl;
	}

	cv::waitKey(0);
	return 0;
}