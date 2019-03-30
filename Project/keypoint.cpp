#include <stdio.h>
#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/stitching.hpp"
#include "opencv2/core/ocl.hpp"

using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;
using namespace cuda;

void readme();

/** @function main */
int main(int argc, char** argv)
{
	/*if (argc != 3)
	{
		return -1;
	}*/



	//Mat img_1 = imread("E:/OneDrive - Arizona State University/MS/DIVP/standard_images/adobe_panoramas/data/carmel/carmel-01.png", IMREAD_GRAYSCALE);
	//Mat img_2 = imread("E:/OneDrive - Arizona State University/MS/DIVP/standard_images/adobe_panoramas/data/carmel/carmel-00.png", IMREAD_GRAYSCALE);

	cv::ocl::setUseOpenCL(false);

	Mat img_1 = imread("./1.jpg");
	Mat img_2 = imread("./2.jpg");

	namedWindow("Matches", CV_WINDOW_FREERATIO);
	namedWindow("Panorama", CV_WINDOW_FREERATIO);

	namedWindow( "bird", cv::WINDOW_AUTOSIZE );
  	imshow( "bird", img_2 );

	bool try_use_gpu = true;
	//Stitcher::Mode mode = Stitcher::PANORAMA;
	Stitcher stitcher = Stitcher::createDefault(true);
	
	vector<Mat> imgs;


	if (!img_1.data || !img_2.data)
	{
		return -1;
	}

	////-- Step 1: Detect the keypoints using SURF Detector, compute the descriptors
	int minHessian = 500;

	Ptr<SURF> detector = SURF::create();
	detector->setHessianThreshold(minHessian);

	std::vector<KeyPoint> keypoints_1, keypoints_2;
	Mat descriptors_1, descriptors_2;

	detector->detectAndCompute(img_1, noArray(), keypoints_1, descriptors_1);
	detector->detectAndCompute(img_2, noArray(), keypoints_2, descriptors_2);

	//-- Step 2: Matching descriptor vectors with a brute force matcher
	BFMatcher matcher(NORM_L2);
	std::vector< DMatch > matches;
	matcher.match(descriptors_1, descriptors_2, matches);

	double max_dist = 0; double min_dist = 100;

	//-- Quick calculation of max and min distances between keypoints
	for (int i = 0; i < descriptors_1.rows; i++)
	{
		double dist = matches[i].distance;
		if (dist < min_dist) min_dist = dist;
		if (dist > max_dist) max_dist = dist;
	}

	printf("-- Max dist : %f \n", max_dist);
	printf("-- Min dist : %f \n", min_dist);

	//-- Draw only "good" matches (i.e. whose distance is less than 3*min_dist )
	std::vector< DMatch > good_matches;

	for (int i = 0; i < descriptors_1.rows; i++)
	{
		if (matches[i].distance < 2*min_dist)
		{
			good_matches.push_back(matches[i]);
		}
	}

	Mat img_matches;
	drawMatches(img_1, keypoints_1, img_2, keypoints_2, good_matches, img_matches);
	for (int i = 0; i < good_matches.size(); i++)
	{
		printf("-- Max dist : %f \n", good_matches[i]);
	}
	
	std::vector<Point2f> obj;
	std::vector<Point2f> scene;
	
	for (int i = 0; i < good_matches.size(); i++)
	{
		//-- Get the keypoints from the good matches
		obj.push_back(keypoints_1[good_matches[i].queryIdx].pt);
		scene.push_back(keypoints_2[good_matches[i].trainIdx].pt);
	}

	Mat H = findHomography(obj, scene, CV_RANSAC);
	
	
	Mat dst;
	warpPerspective(img_2, dst, H, Size(img_1.cols + img_2.cols, img_2.rows), CV_INTER_LINEAR);
	/*
	imshow("Rdst", dst);
	Mat half; 
	half = dst(Rect(0, 0, img_1.cols, img_1.rows));
	//Mat half(dst, Rect(0, 0, img_1.cols, img_1.rows));
	imshow("Half", half);
	imshow("Rdst1", dst);
	cout << half.cols << endl;
	cout << half.rows << endl;
	img_1.copyTo(half(Rect(0, 0, img_1.cols, img_1.rows)));
	imshow("Half_new", half); 
	imshow("dst_new", dst);
	Mat avgBlendResult = dst;
	avgBlendResult.col(img_1.cols) = (0.4*dst.col(img_1.cols) + 0.6*dst.col(img_1.cols + 1));
	Mat roi1 = avgBlendResult(Rect(0, 0, img_2.cols + img_1.cols, img_2.rows));
	imshow("Panorama", roi1); 
	*/
	

	imgs.push_back(img_1);
	imgs.push_back(dst);

	
	Mat pano;
	//Ptr<Stitcher> stitcher = Stitcher::create(mode, try_use_gpu);
	cout << "Here4" << endl;
	/*stitcher.setRegistrationResol(-1); // 0.6
	stitcher.setSeamEstimationResol(-1);   // 0.1
	stitcher.setCompositingResol(-1);   //1
	stitcher.setPanoConfidenceThresh(0.6);   //1
	stitcher.setWaveCorrection(true);
	stitcher.setWaveCorrectKind(detail::WAVE_CORRECT_HORIZ); */
	Stitcher::Status status = stitcher.stitch(imgs, pano);
	//stitcher.stitch(imgs, pano);
	cout << "Here5" << endl;
		//cout << "imgs cols" <<imgs.col<< endl;
		//cout << "imgs rows" <<imgs.row<< endl;
		cout << "pano cols: " <<pano.cols<< endl;
		cout << "pano rows: " <<pano.rows<< endl;
	//imwrite("./matches.jpg", img_matches);
imwrite("./Panorama.jpg", pano);
	//-- Show detected matches
	//imshow("Matches", img_matches);
	//imshow("Panorama Mode", pano);

	waitKey(0);


	return 0;
}

/** @function readme */
void readme()
{
	std::cout << " Usage: ./SURF_descriptor <img1> <img2>" << std::endl;
}
