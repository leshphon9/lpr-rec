#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>
#include <opencv2/cvconfig.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>    

#include <io.h>
#include <string>
#include <iostream>

using namespace cv;
using namespace std;
using namespace ml;


#define CLASSSUM    35   // 图片共有26类
#define IMAGE_ROWS  10   // 统一图片高度
#define IMAGE_COLS  20   // 统一图片宽度
#define IMAGESSUM   50   // 每一类图片张数

string dirNum[CLASSSUM] = { "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "D", "E", "F", "G", "H", "J", \
							"K", "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z" ,"沪" };
//string dirNum[CLASSSUM] = { "0", "1", "2", "3", "4", "5", "6", "7", "8", "9" };
//string dirNum[CLASSSUM] = { "A", "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z" };

float trainingData[CLASSSUM*IMAGESSUM][IMAGE_ROWS*IMAGE_COLS] = { { 0 } };  // 每一行一个训练图片
float labels[CLASSSUM*IMAGESSUM][CLASSSUM] = { { 0 } };                     // 训练样本标签

void TestXml()
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
	// 测试训练结果
	TestXml();
#if 0
	for (int dnum = 0; dnum < CLASSSUM; dnum++)
	{
		int k = 0;
		string inPath = "charSamples/" + dirNum[dnum] + "\\*.png";
		intptr_t handle;
		struct _finddata_t fileinfo;
		handle = _findfirst(inPath.c_str(), &fileinfo);
		if (handle == -1) return -1;

		do {
			string imgname = "charSamples/" + dirNum[dnum] + "/" + fileinfo.name;
			cout << imgname << endl;
			Mat srcImage = imread(imgname, 0);
			if (srcImage.empty()) {
				cout << "Read image error:" << imgname << endl;
				return -1;
			}

			resize(srcImage, srcImage, Size(IMAGE_ROWS, IMAGE_COLS), (0, 0), (0, 0), INTER_AREA);
			threshold(srcImage, srcImage, 0, 255, THRESH_BINARY | THRESH_OTSU);

			for (int j = 0; j < IMAGE_ROWS*IMAGE_COLS; j++) {
				trainingData[dnum*IMAGESSUM + k][j] = (float)srcImage.data[j];
			}

			// 设置标签数据
			for (int j = 0; j < CLASSSUM; j++)
			{
				if (j == dnum)
					labels[dnum*IMAGESSUM + k][j] = 1;
				else
					labels[dnum*IMAGESSUM + k][j] = 0;
			}
			k++;
		} while (!_findnext(handle, &fileinfo) && k < IMAGESSUM);

		Mat labelsMat(CLASSSUM*IMAGESSUM, CLASSSUM, CV_32FC1, labels);
		_findclose(handle);
	}

	// 训练样本数据及对应标签
	Mat trainingDataMat(CLASSSUM*IMAGESSUM, IMAGE_ROWS*IMAGE_COLS, CV_32FC1, trainingData);
	Mat labelsMat(CLASSSUM*IMAGESSUM, CLASSSUM, CV_32FC1, labels);

	// 开始训练
	Ptr<ANN_MLP>model = ANN_MLP::create();
	Mat layerSizes = (Mat_<int>(1, 5) << IMAGE_ROWS * IMAGE_COLS, 128, 128, 128, CLASSSUM);
	model->setLayerSizes(layerSizes);
	model->setTrainMethod(ANN_MLP::BACKPROP, 0.001, 0.1);
	model->setActivationFunction(ANN_MLP::SIGMOID_SYM, 1.0, 1.0);
	model->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER | TermCriteria::EPS, 20000, 0.0001));
	Ptr<TrainData> trainData = TrainData::create(trainingDataMat, ROW_SAMPLE, labelsMat);
	model->train(trainData);
	model->save("ann.xml"); //保存训练结果
	cout << "ann.xml saved" << endl;
#endif
	waitKey(0);

	return 0;
}