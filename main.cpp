#include<opencv2/opencv.hpp>
#include<iostream>

using namespace std;
using namespace cv;

int main(int argc, char** argv) {
				Mat pic = imread("/root/projects/PCA/pic.jpg");//��ȡͼƬ
				namedWindow("origin", WINDOW_AUTOSIZE);
				imshow("origin", pic);
				Mat gray, binary;
				cvtColor(pic, gray, COLOR_BGR2GRAY);//�����ͼƬת��Ϊ�Ҷ�
				threshold(gray, binary, 0, 255, THRESH_BINARY | THRESH_OTSU);//ͼƬ��ֵ��
				imshow("binary", binary);
				vector<vector<Point>>contours;
				vector<Vec4i>hierarchy;
				findContours(binary, contours, hierarchy, RETR_TREE, CHAIN_APPROX_NONE);//�������
				Mat contours_mat;
				Point2f center_point;
				PCA pca;
				for (int i = 0; i < contours.size(); i++) {//��������
								int contours_area = contourArea(contours[i]);
								if (contours_area < 100 || contours_area>100000) {//ɸ�����̫С�Ļ�̫����������õ���Ҫ���������
												continue;
								}
								drawContours(pic, contours, i, Scalar(0, 0, 255), 2, 8);//��������
								contours_mat.create(contours[i].size(), 2, CV_32FC1);//�趨��������ĳߴ�����ݸ�ʽ
								for (int j = 0; j < contours[i].size(); j++) {
												contours_mat.at<float>(j, 0) = contours[i][j].x;//�������ϸ����xy���긳ֵ����������ÿ��Ϊһ����
												contours_mat.at<float>(j, 1) = contours[i][j].y;
								}
								pca(contours_mat, Mat(), PCA::DATA_AS_ROW, 0);//��ÿ�������ĵ����pca���㣬�õ���ֵ��Э����������������
								Point2f center_point(pca.mean.at<float>(0, 0), pca.mean.at<float>(0, 1));//������������ľ�ֵ�㣬��Ϊ���ĵ�
								circle(pic, center_point, 2, Scalar(0, 0, 255), -1);
								Point2f eigen_vector1(80 * pca.eigenvectors.at<float>(0, 0), 80 * pca.eigenvectors.at<float>(0, 1));//eigen_vector1ΪЭ���������������������������������غ�
								Point2f eigen_vector2(20 * pca.eigenvectors.at<float>(1, 0), 20 * pca.eigenvectors.at<float>(1, 1));//eigen_vector2ΪЭ��������������������eigen_vector1����
								line(pic, center_point, center_point + eigen_vector1, Scalar(255, 0, 0), 2);//�������ĵ�Ϊ��㣬eigen_vector1Ϊ�����ֱ��
								line(pic, center_point, center_point + eigen_vector2, Scalar(0, 255, 0), 2);//�������ĵ�Ϊ��㣬eigen_vector2Ϊ�����ֱ��

				}
				imshow("result", pic);
				waitKey(0);

}