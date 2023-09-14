#ifndef IMAGEFEATURES_H
#define IMAGEFEATURES_H 

#include <vector>
#include <opencv2/core.hpp>

struct PointFeature
{
    PointFeature();
    PointFeature(const double & score_, const double & x_, const double & y_);
    double score, x, y;
    bool operator<(const PointFeature & other) const;   // used for std::sort
};

struct ArUco_Details{
    std::vector<std::vector<cv::Point2f>> corners;
    std::vector<int> id;
};

std::vector<PointFeature> detectFeatures(const cv::Mat & img, const int & maxNumFeatures);

ArUco_Details detectArUco(const cv::Mat & img, const int & maxNumFeatures);

#endif
