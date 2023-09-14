#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/aruco.hpp>

#include "imageFeatures.h"

PointFeature::PointFeature()
    : score(0)
    , x(0)
    , y(0)
{}

PointFeature::PointFeature(const double & score_, const double & x_, const double & y_)
    : score(score_)
    , x(x_)
    , y(y_)
{}

bool PointFeature::operator<(const PointFeature & other) const
{
    return (score > other.score);
}

std::vector<PointFeature> detectFeatures(const cv::Mat & img, const int & maxNumFeatures)
{
    std::vector<PointFeature> features;
    // TODO: Lab 8
    // Choose a suitable feature detector
    // Save features above a certain texture threshold
    // Sort features by texture
    // Cap number of features to maxNumFeatures


    cv::Mat imgout = img.clone();
    cv::Mat grayImage;
    cv::cvtColor(img, grayImage, cv::COLOR_BGR2GRAY);

    cv::Mat eigenValues;
    cv::cornerMinEigenVal(grayImage, eigenValues, 3, 3);

    double threshold = 0.001; // Adjust the threshold as needed --> 0.01 is also decent

    std::vector<cv::Point> featurePoints;
    std::vector<double> textureValues;

    int iter=0;
    for (int y = 0; y < eigenValues.rows; y++)
    {
        for (int x = 0; x < eigenValues.cols; x++)
        {
            if (eigenValues.at<float>(y, x) > threshold)
            {
                //cv::circle(imgout, cv::Point(x,y), 5, cv::Scalar(0,0,255), 2, 8, 0 );
                featurePoints.push_back(cv::Point(x, y));
                textureValues.push_back(eigenValues.at<float>(y, x));
                iter++;
            }
        }
    }

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

    int M = std::min(maxNumFeatures, static_cast<int>(indices.size()));

    // Populate the features vector with sorted feature points
    for (int i = 0; i < M; i++)
    {
        double x = sortedFeaturePoints[i].x;
        double y = sortedFeaturePoints[i].y;
        double score = sortedTextureValues[i];

        PointFeature feature(score, x, y);
        features.push_back(feature);
    }


    return features;

}

ArUco_Details detectArUco(const cv::Mat & img, const int & maxNumFeatures){
    cv::aruco::DetectorParameters detectorParams = cv::aruco::DetectorParameters();
    cv::aruco::Dictionary dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);
    cv::aruco::ArucoDetector detector(dictionary, detectorParams);
    
    std::vector<int> ids;
    std::vector<std::vector<cv::Point2f>> corners, rejected;
    detector.detectMarkers(img, corners, ids, rejected);
    
    if (ids.size() > 0)
        cv::aruco::drawDetectedMarkers(img, corners, ids);

        // Flatten the corners vector
    std::vector<cv::Point2f> flattenedCorners;
    for (const auto& cornerSet : corners) {
        for (const auto& corner : cornerSet) {
            flattenedCorners.push_back(corner);
        }
    }

    ArUco_Details result;
    result.corners = corners;
    result.id = ids; 

    return result;
}