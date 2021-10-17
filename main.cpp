#include<opencv2/opencv.hpp>
#include<iostream>

using namespace std;
using namespace cv;

int main(int argc, char** argv) {
				Mat pic = imread("/root/projects/PCA/pic.jpg");//读取图片
				namedWindow("origin", WINDOW_AUTOSIZE);
				imshow("origin", pic);
				Mat gray, binary;
				cvtColor(pic, gray, COLOR_BGR2GRAY);//被检测图片转化为灰度
				threshold(gray, binary, 0, 255, THRESH_BINARY | THRESH_OTSU);//图片二值化
				imshow("binary", binary);
				vector<vector<Point>>contours;
				vector<Vec4i>hierarchy;
				findContours(binary, contours, hierarchy, RETR_TREE, CHAIN_APPROX_NONE);//检测轮廓
				Mat contours_mat;
				Point2f center_point;
				PCA pca;
				for (int i = 0; i < contours.size(); i++) {//遍历轮廓
								int contours_area = contourArea(contours[i]);
								if (contours_area < 100 || contours_area>100000) {//筛掉面积太小的或太大的轮廓，得到需要物体的轮廓
												continue;
								}
								drawContours(pic, contours, i, Scalar(0, 0, 255), 2, 8);//画出轮廓
								contours_mat.create(contours[i].size(), 2, CV_32FC1);//设定轮廓矩阵的尺寸和数据格式
								for (int j = 0; j < contours[i].size(); j++) {
												contours_mat.at<float>(j, 0) = contours[i][j].x;//将轮廓上各点的xy坐标赋值给轮廓矩阵，每行为一个点
												contours_mat.at<float>(j, 1) = contours[i][j].y;
								}
								pca(contours_mat, Mat(), PCA::DATA_AS_ROW, 0);//将每个轮廓的点进行pca计算，得到均值、协方差矩阵的特征向量
								Point2f center_point(pca.mean.at<float>(0, 0), pca.mean.at<float>(0, 1));//画出轮廓坐标的均值点，即为中心点
								circle(pic, center_point, 2, Scalar(0, 0, 255), -1);
								Point2f eigen_vector1(80 * pca.eigenvectors.at<float>(0, 0), 80 * pca.eigenvectors.at<float>(0, 1));//eigen_vector1为协方差矩阵的特性向量，与轮廓跨度最大方向重合
								Point2f eigen_vector2(20 * pca.eigenvectors.at<float>(1, 0), 20 * pca.eigenvectors.at<float>(1, 1));//eigen_vector2为协方差矩阵的特性向量，与eigen_vector1正交
								line(pic, center_point, center_point + eigen_vector1, Scalar(255, 0, 0), 2);//画出中心点为起点，eigen_vector1为方向的直线
								line(pic, center_point, center_point + eigen_vector2, Scalar(0, 255, 0), 2);//画出中心点为起点，eigen_vector2为方向的直线

				}
				imshow("result", pic);
				waitKey(0);

}