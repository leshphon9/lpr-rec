#define img_path "沪KR9888.jpg"
#include "opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"
#include "vector"
#include "iostream"

#define CLASSSUM    35   // 图片共有26类
#define IMAGE_ROWS  10   // 统一图片高度
#define IMAGE_COLS  20   // 统一图片宽度
#define IMAGESSUM   50   // 每一类图片张数
using namespace cv;
using namespace std;
using namespace ml;
string dirNum[CLASSSUM] = { "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "D", "E", "F", "G", "H", "J", \
							"K", "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z" ,"沪" };
vector<vector<Point>> contours;
vector<Vec4i> vec_4f;

void preTreat(Mat& srcImg)
{
	Mat grayImg;
	cvtColor(srcImg, grayImg, COLOR_BGR2GRAY);
	//imshow("grayImg", grayImg);

//gaus
	Mat gausImg;
	GaussianBlur(grayImg, gausImg, Size(3, 3), 0, 0, BORDER_DEFAULT);
	//imshow("gausImg", gausImg);

//sobel
	Mat dst_x, abs_x, sobImg;
	Sobel(gausImg, dst_x, CV_16S, 1, 0);
	convertScaleAbs(dst_x, abs_x);
	sobImg = abs_x;
	//imshow("sobImg", sobImg);

//thres
	Mat thresImg;
	threshold(sobImg, thresImg, 180, 255, THRESH_OTSU);
	//imshow("thresImg", thresImg);

//MORTH
	Mat closeImg, openImg;
	Mat elementX = getStructuringElement(MORPH_RECT, Size(30, 30)), elementY;
	morphologyEx(thresImg, closeImg, MORPH_CLOSE, elementX);
	//morphologyEx(closeImg, openImg, MORPH_OPEN, elementX);
	elementX = getStructuringElement(MORPH_RECT, Size(15, 3));//20 1
	elementY = getStructuringElement(MORPH_RECT, Size(3, 15));//1 20
	dilate(closeImg, closeImg, elementX);
	erode(closeImg, closeImg, elementX);
	erode(closeImg, closeImg, elementY);
	dilate(closeImg, closeImg, elementY);
	//imshow("closeImg", closeImg);
//Contours
	findContours(closeImg, contours, vec_4f, RETR_TREE, CHAIN_APPROX_SIMPLE);
	//drawContours(srcImg, contours, -1, Scalar(0, 255, 0), 1);
	//imshow("srcImg", srcImg);
}

Mat PRrect(Mat& srcImg)
{
	Mat rectImg, grayRI, gausRI, thresRI;
	for (int i = 0; i < contours.size(); i++) {
		Rect rect = boundingRect(contours[i]);
		int x = rect.x;
		int y = rect.y;
		if ((rect.width > (rect.height * 2)) && (rect.width < (rect.height * 5)))
		{
			rectImg = srcImg(Rect(rect.x, rect.y, rect.width, rect.height));
			//cvtColor(rectImg, grayRI, COLOR_BGR2GRAY);
			//GaussianBlur(grayRI, gausRI, Size(3, 3), 0, 0, BORDER_DEFAULT);
			//threshold(gausRI, thresRI, 180, 255, THRESH_OTSU);
			//rectangle(srcImg, Point(rect.x, rect.y), Point(rect.x + rect.width, rect.y + rect.height), Scalar(0, 0, 255), 2); 
		}
	}
	//imshow("srcImg", srcImg);
	imshow("rectImg", rectImg);
	return rectImg;
}

void split(Mat& thresRI,Mat& rectImg)
{
	Mat closeRI;

	Mat ele = getStructuringElement(MORPH_RECT, Size(10, 10));
	morphologyEx(thresRI, closeRI, MORPH_CLOSE, ele);
	//imshow("closeRI", closeRI);
	Mat eleX = getStructuringElement(MORPH_RECT, Size(5, 5));
	Mat eleY = getStructuringElement(MORPH_RECT, Size(5, 5));
	dilate(closeRI, closeRI, eleX);
	erode(closeRI, closeRI, eleX);
	erode(closeRI, closeRI, eleY);
	dilate(closeRI, closeRI, eleY);

	Canny(closeRI, closeRI, 300, 300);
	//imshow("closeRI", closeRI);

	vector<vector<Point>> contours2;
	vector<Vec4i> hierarchy;
	findContours(closeRI, contours2, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_NONE, Point());
	int size = (int)(contours2.size());
	// 保存符号边框的序号
	vector<int> num_order;
	map<int, int> num_map;
	for (int i = 0; i < size; i++) {
		// 获取边框数据
		Rect number_rect = boundingRect(contours2[i]);
		int width = number_rect.width;
		int height = number_rect.height;
		// 去除较小的干扰边框，筛选出合适的区域
		if (width > thresRI.cols / 10 && height > thresRI.rows / 2) {
			//rectangle(thresRI, number_rect.tl(), number_rect.br(), Scalar(255, 255, 255), 1, 1, 0);
			//drawContours(rectImg, contours2, -1, Scalar(0, 0, 255), 1);
			//imshow("rectImg", rectImg);
			num_order.push_back(number_rect.x);
			num_map[number_rect.x] = i;
		}
	}
	// 按符号顺序提取
	sort(num_order.begin(), num_order.end());
	for (int i = 0; i < num_order.size(); i++) {
		Rect number_rect = boundingRect(contours2[num_map.find(num_order[i])->second]);
		Rect choose_rect(number_rect.x, 0, number_rect.width, rectImg.rows);
		Mat number_img = rectImg(choose_rect);
		imshow("number" + to_string(i), number_img);
		//imwrite("number" + to_string(i) + ".jpg", number_img);
	}
}

void rec()
{
	Ptr<ANN_MLP> model = StatModel::load<ANN_MLP>("ann.xml");
	double maxVal = 0;
	Point maxLoc;
	for (int i = 0; i <= 6; i++)
	{
		//string imgPath = "number" + i;
		string inPath = "number" + dirNum[i] + ".jpg";
		//Mat srcImage = imread("pr-test/Y.jpg", IMREAD_GRAYSCALE);
		Mat srcImage = imread(inPath, IMREAD_GRAYSCALE);

		if (!srcImage.empty())
		{
			int img_r = srcImage.rows;
			int img_c = srcImage.cols;
			//将测试图像转化为1*128的向量
			resize(srcImage, srcImage, Size(IMAGE_ROWS, IMAGE_COLS), (0, 0), (0, 0), INTER_AREA);
			threshold(srcImage, srcImage, 0, 255, THRESH_BINARY | THRESH_OTSU);

			Mat_<float> testMat(1, IMAGE_ROWS*IMAGE_COLS);

			for (int i = 0; i < IMAGE_ROWS*IMAGE_COLS; i++)
			{
				testMat.at<float>(0, i) = (float)srcImage.at<uchar>(i / IMAGE_ROWS, i % IMAGE_COLS);
			}

			//使用训练好的MLP model预测测试图像
			Mat dst;
			model->predict(testMat, dst);
			minMaxLoc(dst, NULL, &maxVal, NULL, &maxLoc);
		}
		//imshow("srcimg", srcImage);
		cout << dirNum[maxLoc.x];
		//cout << inPath << endl << "测试结果：" << dirNum[maxLoc.x] << endl << "置信度:" << maxVal * 100 << "%" << endl << endl;
	}
}

int main()
{
	Mat srcImg = imread(img_path);
	//imshow("srcImg", srcImg);
	Mat srcImgB = srcImg.clone();
	preTreat(srcImgB);
	Mat rectImg = PRrect(srcImgB);

	
	Mat grayRI, gausRI, thresRI;
	cvtColor(rectImg, grayRI, COLOR_BGR2GRAY);
	GaussianBlur(grayRI, gausRI, Size(3, 3), 0, 0, BORDER_DEFAULT);
	threshold(gausRI, thresRI, 180, 255, THRESH_OTSU);
	//imshow("thresRI", thresRI);
	split(thresRI, rectImg);
	rec();
	
	waitKey(0);
	return 0;
}

