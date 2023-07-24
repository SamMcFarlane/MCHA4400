#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/aruco.hpp>
#include "imagefeatures.h"
#include <numeric>
#include <iostream>

cv::Mat detectAndDrawHarris(const cv::Mat & img, int maxNumFeatures)
{
    cv::Mat imgout = img.clone();

    // TODO
    cv::Mat grayImage;
    cv::cvtColor(img, grayImage, cv::COLOR_BGR2GRAY);

    cv::Mat dst;
    //cv::Mat dst_cpy;
    cv::cornerHarris(grayImage,dst,2,3,0.04);
    //dst_cpy = dst;
    double threshold = 150;

    
    cv::Mat dst_norm, dst_norm_scaled;
    normalize( dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat() );
    convertScaleAbs( dst_norm, dst_norm_scaled );

    std::vector<cv::Point> featurePoints;
    std::vector<double> textureValues; 

    for(int i =0;i<dst_norm.rows;i++){
        for(int j =0;j<dst_norm.cols;j++){
            if((int) dst_norm.at<float>(i,j) > threshold){
               cv::circle(imgout, cv::Point(j,i), 5, cv::Scalar(0,0,255), 2, 8, 0 );
                featurePoints.push_back(cv::Point(j, i));
                textureValues.push_back(dst.at<float>(i, j));
            }

        }
    }

    std::vector<int> indices(textureValues.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(), [&](int a, int b) { return textureValues[a] > textureValues[b]; });

std::cout<<"Using Harris Corner Detector"<<std::endl;
std::cout<<"Image Width: "<<std::to_string(img.cols)<<std::endl;
std::cout<<"Image Height: "<<std::to_string(img.rows)<<std::endl;
std::cout<<"Features Requested: "<<std::to_string(maxNumFeatures)<<std::endl;
std::cout<<"Features Found: "<<textureValues.size()<<std::endl;

int featuresFound = (int) textureValues.size();

 for (int i = 0; i < std::min(maxNumFeatures,featuresFound); i++)
    {
        cv::Point point = featurePoints[indices[i]];
        cv::Scalar color(0, 255, 255); // Red color for the circle

        // Draw a circle around the feature point
        int radius = 5;
        int thickness = 2;
        cv::circle(imgout, point, radius, color, thickness);

        // Print the Harris score of the feature point
        std::string scoreStr = std::to_string(textureValues[indices[i]]);
        cv::putText(imgout, scoreStr, point, cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 0, 0),2);

        std::cout<<"idx: "<<i<<"  | Pt: ("<<point.x<<","<<point.y<<")  | Harris Score: "<<scoreStr<<std::endl;
    }


    return imgout;
}

cv::Mat detectAndDrawShiAndTomasi(const cv::Mat & img, int maxNumFeatures)
{
    cv::Mat imgout = img.clone();

    // TODO

    // TODO
    cv::Mat grayImage;
    cv::cvtColor(img, grayImage, cv::COLOR_BGR2GRAY);

    // Step 1: Use Shi & Tomasi corner detector to find corners of interest
    cv::Mat eigenValues;
    cv::cornerMinEigenVal(grayImage, eigenValues, 3, 3);

    double threshold = 0.01; // Adjust the threshold as needed

    // Step 2: Store all pixels and their associated eigenvalues above the threshold in a vector
    std::vector<cv::Point> featurePoints;
    std::vector<double> textureValues;

    int iter=0;
    for (int y = 0; y < eigenValues.rows; y++)
    {
        for (int x = 0; x < eigenValues.cols; x++)
        {
            if (eigenValues.at<float>(y, x) > threshold)
            {
                cv::circle(imgout, cv::Point(x,y), 5, cv::Scalar(0,0,255), 2, 8, 0 );
                featurePoints.push_back(cv::Point(x, y));
                textureValues.push_back(eigenValues.at<float>(y, x));
                iter++;
            }
        }
    }

    // Step 3: Sort the texture values and feature points in descending order
    std::vector<int> indices(textureValues.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(), [&](int a, int b) { return textureValues[a] > textureValues[b]; });

    int N = std::min(maxNumFeatures, static_cast<int>(indices.size()));
    std::vector<cv::Point> sortedFeaturePoints;
    std::vector<double> sortedTextureValues;

    for (int i = 0; i < N; i++)
    {
        sortedFeaturePoints.push_back(featurePoints[indices[i]]);
        sortedTextureValues.push_back(textureValues[indices[i]]);
    }

    featurePoints = sortedFeaturePoints;
    textureValues = sortedTextureValues;

    std::cout<<"Using Shi and Tomasi Corner Detector"<<std::endl;
    std::cout<<"Image Width: "<<std::to_string(img.cols)<<std::endl;
    std::cout<<"Image Height: "<<std::to_string(img.rows)<<std::endl;
    std::cout<<"Features Requested: "<<std::to_string(maxNumFeatures)<<std::endl;
    std::cout<<"Features Found: "<<iter<<std::endl;

    int featuresFound = (int) textureValues.size();

   for (int i = 0; i < std::min(maxNumFeatures,featuresFound); i++)
    {
        cv::Point point = featurePoints[indices[i]];
        cv::Scalar color(0, 255, 255); // Red color for the circle

        // Draw a circle around the feature point
        int radius = 5;
        int thickness = 2;
        cv::circle(imgout, point, radius, color, thickness);

        // Print the Harris score of the feature point
        std::string scoreStr = std::to_string(textureValues[indices[i]]);
        cv::putText(imgout, scoreStr, point, cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 0, 0),2);

        std::cout<<"idx: "<<i<<"  | Pt: ("<<point.x<<","<<point.y<<")  | Harris Score: "<<scoreStr<<std::endl;
    }


    return imgout;
}

cv::Mat detectAndDrawORB(const cv::Mat & img, int maxNumFeatures)
{
    cv::Mat imgout = img.clone();

    cv::Mat grayImage;
    cv::cvtColor(img, grayImage, cv::COLOR_BGR2GRAY);

    float scaleFactor = 1.2f;
    int nlevels = 8;
    int edgeThreshold = 25;
    int firstLevel = 0;
    int WTA_K = 2;
    cv::ORB::ScoreType scoreType = cv::ORB::HARRIS_SCORE;
    int patchSize = 25;
    int fastThreshold = 40;


    cv::Ptr<cv::ORB> orb = cv::ORB::create(maxNumFeatures,scaleFactor,nlevels,edgeThreshold,firstLevel,WTA_K,scoreType,patchSize,fastThreshold);

    std::vector<cv::KeyPoint> keypoints;

    orb->detect(grayImage,keypoints);

    cv::Mat descriptors;
    orb->compute(grayImage, keypoints, descriptors);

    // Draw keypoints on the image
    cv::Mat image_with_keypoints;
    cv::drawKeypoints(img, keypoints, imgout, cv::Scalar(0, 0, 255), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

    std::cout<<"Using ORB Detector"<<std::endl;
    std::cout<<"Image Width: "<<std::to_string(img.cols)<<std::endl;
    std::cout<<"Image Height: "<<std::to_string(img.rows)<<std::endl;
    std::cout<<"Descriptor Width: "<<std::to_string(descriptors.cols)<<std::endl;
    std::cout<<"Descripter Height: "<<descriptors.rows<<std::endl;

    for(int i = 0; i<((int) keypoints.size());i++){
        std::cout<<"KeyPoint "<<i<<"  | Descriptors  [";
        for(int j=0;j<descriptors.cols;j++){
            uchar value = descriptors.at<uchar>(i,j);
            std::cout<<static_cast<int>(value)<<" ";
        }
        std::cout<<"]"<<std::endl;
    }

    
    return imgout;
}

cv::Mat detectAndDrawArUco(const cv::Mat & img, int maxNumFeatures)
{
    cv::Mat imgout = img.clone();

    // TODO
    cv::aruco::DetectorParameters detectorParams = cv::aruco::DetectorParameters();
    cv::aruco::Dictionary dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);
    cv::aruco::ArucoDetector detector(dictionary, detectorParams);
    
    std::vector<int> ids;
    std::vector<std::vector<cv::Point2f>> corners, rejected;
    detector.detectMarkers(img, corners, ids, rejected);
    
    if (ids.size() > 0)
        cv::aruco::drawDetectedMarkers(imgout, corners, ids);
    else{
        std::cout<<"No tags found"<<std::endl;
    }

    std::cout<<"Using ArUco"<<std::endl;
    std::cout<<"Image Width: "<<std::to_string(img.cols)<<std::endl;
    std::cout<<"Image Height: "<<std::to_string(img.rows)<<std::endl;
    std::cout<<"Number of Tages: "<<(int) ids.size()<<std::endl;

    for(int i = 0;i<(int) ids.size();i++){
        std::cout<<"ID: "<<ids.at(i)<<"  |  Corners: ";
        for(int j = 0;j<4;j++){
           cv::Point2f point = corners[i][j];
            std::cout<<"("<<point.x<<","<<point.y<<") ";
        }
        std::cout<<std::endl;
    }


    return imgout;
}