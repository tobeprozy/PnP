#include<opencv2/opencv.hpp>
#include<iostream>
#include <Eigen/Dense>
#include <Eigen/Core>

using namespace cv;
using namespace std;
//Eigen::Isometry3d Get3DR_TransMatrix(const std::vector<Eigen::Vector3d>& srcPoints, const std::vector<Eigen::Vector3d>& dstPoints);
cv::Mat Get3DR_TransMatrix(const std::vector<cv::Point3f>& srcPoints, const std::vector<cv::Point3f>& dstPoints);
int main()
{
	std::vector<cv::Point3f> srcPoints;
	std::vector<cv::Point3f>  dstPoints;

	//取三组点
	float NN = 100;
	srcPoints.push_back(cv::Point3f(249.6535873, -18.88801336, 333.7594986));
	dstPoints.push_back(cv::Point3f(-265.717, -22.8008, -236.5));

	srcPoints.push_back(cv::Point3f(191.4942169, -7.847265601, 322.7531052));
	dstPoints.push_back(cv::Point3f(-289.395, -21.0469, -178));

	srcPoints.push_back(cv::Point3f(184.4823074, 11.78870082, 312.0020676));
	dstPoints.push_back(cv::Point3f(-311.318, -29.8164, -175.5));

	cv::Mat RT = Get3DR_TransMatrix(srcPoints, dstPoints);
	for (int r = 0; r < RT.rows; r++)
	{
		for (int c = 0; c < RT.cols; c++)
		{
			printf("%f, ", RT.at<double>(r, c));
		}
		printf("\n");
	}
	printf("**************************************\n");
	getchar();
}

cv::Mat Get3DR_TransMatrix(const std::vector<cv::Point3f>& srcPoints, const std::vector<cv::Point3f>& dstPoints)
{
	double srcSumX = 0.0f;
	double srcSumY = 0.0f;
	double srcSumZ = 0.0f;

	double dstSumX = 0.0f;
	double dstSumY = 0.0f;
	double dstSumZ = 0.0f;

	//至少三组点
	if (srcPoints.size() != dstPoints.size() || srcPoints.size() < 3)
	{
		return cv::Mat();
	}

	int pointsNum = srcPoints.size();
	for (int i = 0; i < pointsNum; ++i)
	{
		srcSumX += srcPoints[i].x;
		srcSumY += srcPoints[i].y;
		srcSumZ += srcPoints[i].z;

		dstSumX += dstPoints[i].x;
		dstSumY += dstPoints[i].y;
		dstSumZ += dstPoints[i].z;
	}

	cv::Point3d centerSrc, centerDst;

	centerSrc.x = double(srcSumX / pointsNum);
	centerSrc.y = double(srcSumY / pointsNum);
	centerSrc.z = double(srcSumZ / pointsNum);

	centerDst.x = double(dstSumX / pointsNum);
	centerDst.y = double(dstSumY / pointsNum);
	centerDst.z = double(dstSumZ / pointsNum);

	//Mat::Mat(int rows, int cols, int type)
	cv::Mat srcMat(3, pointsNum, CV_64FC1);
	cv::Mat dstMat(3, pointsNum, CV_64FC1);

	for (int i = 0; i < pointsNum; ++i)//N组点
	{
		//三行
		srcMat.at<double>(0, i) = srcPoints[i].x - centerSrc.x;
		srcMat.at<double>(1, i) = srcPoints[i].y - centerSrc.y;
		srcMat.at<double>(2, i) = srcPoints[i].z - centerSrc.z;

		dstMat.at<double>(0, i) = dstPoints[i].x - centerDst.x;
		dstMat.at<double>(1, i) = dstPoints[i].y - centerDst.y;
		dstMat.at<double>(2, i) = dstPoints[i].z - centerDst.z;

	}

	cv::Mat matS = srcMat * dstMat.t();

	cv::Mat matU, matW, matV;
	cv::SVDecomp(matS, matW, matU, matV);

	cv::Mat matTemp = matU * matV;
	double det = cv::determinant(matTemp);//行列式的值

	double datM[] = { 1, 0, 0, 0, 1, 0, 0, 0, det };
	cv::Mat matM(3, 3, CV_64FC1, datM);


	for (int r = 0; r < matM.rows; r++)
	{
		for (int c = 0; c < matM.cols; c++)
		{
			printf("%f, ", matM.at<double>(r, c));
		}
		printf("\n");
	}
	cv::Mat matR = matV.t() * matM * matU.t();

	double* datR = (double*)(matR.data);
	double delta_X = centerDst.x - (centerSrc.x * datR[0] + centerSrc.y * datR[1] + centerSrc.z * datR[2]);
	double delta_Y = centerDst.y - (centerSrc.x * datR[3] + centerSrc.y * datR[4] + centerSrc.z * datR[5]);
	double delta_Z = centerDst.z - (centerSrc.x * datR[6] + centerSrc.y * datR[7] + centerSrc.z * datR[8]);


	//生成RT齐次矩阵(4*4)
	cv::Mat R_T = (cv::Mat_<double>(4, 4) <<
		matR.at<double>(0, 0), matR.at<double>(0, 1), matR.at<double>(0, 2), delta_X,
		matR.at<double>(1, 0), matR.at<double>(1, 1), matR.at<double>(1, 2), delta_Y,
		matR.at<double>(2, 0), matR.at<double>(2, 1), matR.at<double>(2, 2), delta_Z,
		0, 0, 0, 1
		);

	return R_T;
}



//
//Eigen::Isometry3d Get3DR_TransMatrix(const std::vector<Eigen::Vector3d>& srcPoints, const std::vector<Eigen::Vector3d>& dstPoints)
//{
//	Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
//
//	double srcSumX = 0.0f;
//	double srcSumY = 0.0f;
//	double srcSumZ = 0.0f;
//
//	double dstSumX = 0.0f;
//	double dstSumY = 0.0f;
//	double dstSumZ = 0.0f;
//
//	//至少三组点
//	if (srcPoints.size() != dstPoints.size() || srcPoints.size() < 3)
//	{
//		return T;
//	}
//	int pointsNum = srcPoints.size();
//	for (int i = 0; i < pointsNum; ++i)
//	{
//		srcSumX += srcPoints[i].x;
//		srcSumY += srcPoints[i].y;
//		srcSumZ += srcPoints[i].z;
//
//		dstSumX += dstPoints[i].x;
//		dstSumY += dstPoints[i].y;
//		dstSumZ += dstPoints[i].z;
//	}
//	Eigen::Vector3d centerSrc, centerDst;
//
//	centerSrc.x = double(srcSumX / pointsNum);
//	centerSrc.y = double(srcSumY / pointsNum);
//	centerSrc.z = double(srcSumZ / pointsNum);
//
//	centerDst.x = double(dstSumX / pointsNum);
//	centerDst.y = double(dstSumY / pointsNum);
//	centerDst.z = double(dstSumZ / pointsNum);
//
//	Eigen::MatrixX3d  srcMat;
//	Eigen::MatrixX3d  dstMat;
//
//
//	for (int i = 0; i < pointsNum; ++i)//N组点
//	{
//		//三行
//		srcMat(0, i) = srcPoints[i].x - centerSrc.x;
//		srcMat(1, i) = srcPoints[i].y - centerSrc.y;
//		srcMat(2, i) = srcPoints[i].z - centerSrc.z;
//
//		dstMat(0, i) = dstPoints[i].x - centerDst.x;
//		dstMat(1, i) = dstPoints[i].y - centerDst.y;
//		dstMat(2, i) = dstPoints[i].z - centerDst.z;
//
//	}
//
//	Eigen::Matrix3d   matS = srcMat * dstMat.transpose();
//	Eigen::JacobiSVD<Eigen::MatrixXf> svd(matS, Eigen::ComputeThinU | Eigen::ComputeThinV);
//
//	Eigen::Matrix3d matV = svd.matrixV(), matU = svd.matrixU();
//
//	Eigen::Matrix3d matM;
//	matM << 1, 0, 0, 0, 1, 0, 0, 0, matU* matV.determinant();
//	Eigen::Matrix3d matR = matV.transpose() * matM * matU.transpose();
//
//
//	double delta_X = centerDst.x - (centerSrc.x * matR(0, 0) + centerSrc.y * matR(0, 1) + centerSrc.z * matR(0, 2));
//	double delta_Y = centerDst.y - (centerSrc.x * matR(1, 0) + centerSrc.y * matR(1, 1) + centerSrc.z * matR(1, 2));
//	double delta_Z = centerDst.z - (centerSrc.x * matR(2, 0) + centerSrc.y * matR(2, 1) + centerSrc.z * matR(2, 2));
//	Eigen::Vector3d  t(delta_X, delta_Y, delta_Z);
//	T.rotate(matR);
//	T.translate(t);
//	return T;
//
//}
