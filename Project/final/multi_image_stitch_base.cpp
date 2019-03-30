#include <stdio.h>
#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/stitching.hpp"


using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;
using namespace cuda;

int main()
{
	 // input file names (for example 1.jpg, 2.jpg etc) t store in the vector for image stitching */
	vector<String> filenames;
	String folder = "./data8"; // again we are using the Opencv's embedded "String" class
	vector<Mat> img;
	Mat image;
	glob(folder, filenames); // new function that does the job ;-)

	for (size_t i = 0; i < filenames.size(); ++i)
	{
		Mat src = imread(filenames[i], CV_LOAD_IMAGE_UNCHANGED);
		img.push_back(src);
		if (!src.data)
			cerr << "Problem loading image!!!" << endl;

		/* do whatever you want with your images here */
	}
	Mat image1, image2;
	image2 = img[0];
	Size size(512, 512);
	for (int k = 0; k < img.size()-1; k++)
	{
		image1 = img[k+1];

		resize(image1, image1, size);
		resize(image2, image2, size);

		/*Convert to gray to extract descriptors*/
		Mat gray_image1;
		Mat gray_image2;

		if (image1.channels() != 3 && image2.channels() != 3)
		{
			gray_image1 = image1;
			gray_image2 = image2;
		}
		else
		{
			//Convert to Grayscale
			cvtColor(image1, gray_image1, CV_RGB2GRAY);
			cvtColor(image2, gray_image2, CV_RGB2GRAY);
		}
		

		if (!gray_image1.data || !gray_image2.data)
		{
			std::cout << " --(!) Error reading images " << std::endl;
			return -1;
		}

		//--Step 1 : Detect the keypoints using SURF Detector

		int minHessian = 400;
		//Ptr<SURF> detector = SURF::create();
		Ptr<ORB> detector = ORB::create();
		//detector->setHessianThreshold(minHessian);

		std::vector< KeyPoint > keypoints_object, keypoints_scene;

		detector->detect(gray_image1, keypoints_object);
		detector->detect(gray_image2, keypoints_scene);

		//--Step 2 : Calculate Descriptors (feature vectors)
		Mat descriptors_object, descriptors_scene;

		detector->compute(gray_image1, keypoints_object, descriptors_object);
		detector->compute(gray_image2, keypoints_scene, descriptors_scene);

		descriptors_scene.convertTo(descriptors_scene, CV_32F);
		descriptors_object.convertTo(descriptors_object, CV_32F);


		//--Step 3 : Matching descriptor vectors using FLANN matcher
		FlannBasedMatcher matcher;
		//FlannBasedMatcher matcher(new flann::LshIndexParams(20, 10, 2));
		//BFMatcher matcher;
		std::vector< DMatch > matches;
		matcher.match(descriptors_object, descriptors_scene, matches);

		double max_dist = 0;
		double min_dist = 100;

		//--Quick calculation of min-max distances between keypoints
		for (int i = 0; i < descriptors_object.rows; i++)
		{

			double dist = matches[i].distance;

			if (dist < min_dist) min_dist = dist;
			if (dist > max_dist) max_dist = dist;
		}

		printf("-- Max dist : %f \n", max_dist);
		printf("-- Min dist : %f \n", min_dist);

		//--Use only "good" matches (i.e. whose distance is less than 3 X min_dist )
		std::vector< DMatch > good_matches;

		for (int i = 0; i < descriptors_object.rows; i++)
		{
			if (matches[i].distance < 3 * min_dist)
			{
				good_matches.push_back(matches[i]);
			}
		}

		std::vector< Point2f > obj;
		std::vector< Point2f > scene;

		for (int i = 0; i < good_matches.size(); i++)
		{
			//--Get the keypoints from the good matches
			obj.push_back(keypoints_object[good_matches[i].queryIdx].pt);
			scene.push_back(keypoints_scene[good_matches[i].trainIdx].pt);
		}

		//Find the Homography Matrix
		Mat H = findHomography(obj, scene, CV_RANSAC, 3);

		// Use the homography Matrix to warp the images
		cv::Mat result;

cout<<"image1.rows : "<<image1.rows<<endl;
cout<<"image1.cols : "<<image1.cols <<endl;
cout<<"image2.cols : "<<image2.cols<<endl;
cout<<"image2.rows : "<<image1.rows<<endl;
cout<<" H.rows : "<<H.cols<<endl;
cout<<" H.cols : "<<H.cols<<endl;
	//if (!H.empty())
	//{
		warpPerspective(image1, result, H, Size(image1.cols + image2.cols, image1.rows));
	

cout<<"result.rows : "<<result.rows<<endl;
cout<<"result.cols : "<<result.cols <<endl;
	
		Mat half;
		half = result(Rect(0, 0, image2.cols, image2.rows));
cout<<"OK 1"<<endl;
		image2.copyTo(half);
		/* To remove the black portion after stitching, and confine in a rectangular region*/
cout<<"OK 2"<<endl;
		// vector with all non-black point positions
		std::vector<cv::Point> nonBlackList;
		nonBlackList.reserve(result.rows*result.cols);
cout<<"OK 3"<<endl;
		// add all non-black points to the vector
		// there are more efficient ways to iterate through the image

		for (int j = 0; j<result.rows; j++)
			for (int i = 0; i<result.cols; i++)
			{
				// if not black: add to the list
				//if (result.at<Vec3b>(j, i) != Vec3b(0,0,0))    //For colour images
				
				if (image1.channels() != 3 && image2.channels() != 3)
				{
					if (result.at<uint8_t>(j, i) != 0)
					{
						nonBlackList.push_back(Point(i, j));
					}
					
				}
				else
				{
					if (result.at<Vec3b>(j, i) != Vec3b(0, 0, 0))
					{
						nonBlackList.push_back(Point(i, j));
					}
				}
				
			}
		// create bounding rect around those points
		Rect bb = cv::boundingRect(nonBlackList);
		image2 = result(bb);

//}


		if (img.size() < 3)
			break;
		else
			image1 = img[k];
		
	}
	imshow("Panorama", image2);
	imwrite("E:/image.jpg", image2);
	waitKey();
	return 0;
}
