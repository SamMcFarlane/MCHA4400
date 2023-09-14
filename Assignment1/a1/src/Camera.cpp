#include <cassert>
#include <cstddef>
#include <cmath>
#include <limits>
#include <iostream>
#include <vector>
#include <filesystem>
#include <regex>
#include <Eigen/Core>
#include <opencv2/core/eigen.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/persistence.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/calib3d.hpp>
#include "rotation.hpp"
#include "Camera.h"

void Chessboard::write(cv::FileStorage& fs) const
{
    fs << "{"
       << "grid_width"  << boardSize.width
       << "grid_height" << boardSize.height
       << "square_size" << squareSize
       << "}";
}

void Chessboard::read(const cv::FileNode& node)
{
    node["grid_width"]  >> boardSize.width;
    node["grid_height"] >> boardSize.height;
    node["square_size"] >> squareSize;
}

std::vector<cv::Point3f> Chessboard::gridPoints() const
{
    std::vector<cv::Point3f> rPNn_all;
    rPNn_all.reserve(boardSize.height*boardSize.width);
    for (int i = 0; i < boardSize.height; ++i)
        for (int j = 0; j < boardSize.width; ++j)
            rPNn_all.push_back(cv::Point3f(j*squareSize, i*squareSize, 0));   
    return rPNn_all; 
}

std::ostream & operator<<(std::ostream & os, const Chessboard & chessboard)
{
    return os << "boardSize: " << chessboard.boardSize << ", squareSize: " << chessboard.squareSize;
}

ChessboardImage::ChessboardImage(const cv::Mat & image_, const Chessboard & chessboard, const std::filesystem::path & filename_)
    : image(image_)
    , filename(filename_)
    , isFound(false)
{
     // TODO:
    //  - Detect chessboard corners in image and set the corners member
    //  - (optional) Do subpixel refinement of detected corners
    // Convert image to grayscale for corner detection
    cv::Mat grayImage;
    cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);
    
    // Detect the corners in the image
    isFound = cv::findChessboardCorners(grayImage, chessboard.boardSize, corners, 
        cv::CALIB_CB_ADAPTIVE_THRESH + cv::CALIB_CB_NORMALIZE_IMAGE + cv::CALIB_CB_FAST_CHECK);

    if (isFound)
    {
        // Optional: refine the corner locations at the subpixel level
        cv::TermCriteria criteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.1);
        cv::cornerSubPix(grayImage, corners, cv::Size(5,5), cv::Size(-1,-1), criteria);
    }
}

void ChessboardImage::drawCorners(const Chessboard & chessboard)
{
    cv::drawChessboardCorners(image, chessboard.boardSize, corners, isFound);
}

void ChessboardImage::drawBox(const Chessboard & chessboard, const Camera & camera)
{
     // TODO

       // Get the dimensions of the chessboard
    int boardWidth = chessboard.boardSize.width;
    int boardHeight = chessboard.boardSize.height;
    float squareSize = chessboard.squareSize;

    // Height of the rectangular prism
    double height = -0.23;

    //this->recoverPose(chessboard,camera);


    for(int i = 0;i<4*(boardWidth-1);i++){

        cv::Vec3d bottomS1c(squareSize * i/4,0,0);
        cv::Vec3d bottomS1n(squareSize * (i+1)/4,0,0);

        cv::Vec3d bottomS2c(squareSize * i/4,squareSize * (boardHeight-1),0);
        cv::Vec3d bottomS2n(squareSize * (i+1)/4,squareSize * (boardHeight-1),0);

        cv::Vec2d bottomS1current = camera.worldToPixel(bottomS1c, cameraPose);
        cv::Vec2d bottomS1next = camera.worldToPixel(bottomS1n, cameraPose);
        cv::Vec2d bottomS2current = camera.worldToPixel(bottomS2c, cameraPose);
        cv::Vec2d bottomS2next = camera.worldToPixel(bottomS2n, cameraPose);

        cv::Point bottomS1currentPoint(bottomS1current[0],bottomS1current[1]);
        cv::Point bottomS1nextPoint(bottomS1next[0],bottomS1next[1]);
        cv::Point bottomS2currentPoint(bottomS2current[0],bottomS2current[1]);
        cv::Point bottomS2nextPoint(bottomS2next[0],bottomS2next[1]);

        cv::line(image,bottomS1currentPoint,bottomS1nextPoint, cv::Scalar(255, 0, 0), 3);
        cv::line(image,bottomS2currentPoint,bottomS2nextPoint, cv::Scalar(255, 0, 0), 3);
    }

    for(int i = 0;i<4*(boardHeight-1);i++){

        cv::Vec3d bottomS1c(0,squareSize * i/4,0);
        cv::Vec3d bottomS1n(0,squareSize * (i+1)/4,0);

        cv::Vec3d bottomS2c(squareSize * (boardWidth-1),squareSize * i/4,0);
        cv::Vec3d bottomS2n(squareSize * (boardWidth-1),squareSize * (i+1)/4,0);

        cv::Vec2d bottomS1current = camera.worldToPixel(bottomS1c, cameraPose);
        cv::Vec2d bottomS1next = camera.worldToPixel(bottomS1n, cameraPose);
        cv::Vec2d bottomS2current = camera.worldToPixel(bottomS2c, cameraPose);
        cv::Vec2d bottomS2next = camera.worldToPixel(bottomS2n, cameraPose);

        cv::Point bottomS1currentPoint(bottomS1current[0],bottomS1current[1]);
        cv::Point bottomS1nextPoint(bottomS1next[0],bottomS1next[1]);
        cv::Point bottomS2currentPoint(bottomS2current[0],bottomS2current[1]);
        cv::Point bottomS2nextPoint(bottomS2next[0],bottomS2next[1]);

        cv::line(image,bottomS1currentPoint,bottomS1nextPoint, cv::Scalar(0, 255, 0), 3);
        cv::line(image,bottomS2currentPoint,bottomS2nextPoint, cv::Scalar(0, 255, 0), 3);
    }


 for(int i = 0;i<4*(boardWidth-1);i++){

        cv::Vec3d bottomS1c(squareSize * i/4,0,height);
        cv::Vec3d bottomS1n(squareSize * (i+1)/4,0,height);

        cv::Vec3d bottomS2c(squareSize * i/4,squareSize * (boardHeight-1),height);
        cv::Vec3d bottomS2n(squareSize * (i+1)/4,squareSize * (boardHeight-1),height);

        cv::Vec2d bottomS1current = camera.worldToPixel(bottomS1c, cameraPose);
        cv::Vec2d bottomS1next = camera.worldToPixel(bottomS1n, cameraPose);
        cv::Vec2d bottomS2current = camera.worldToPixel(bottomS2c, cameraPose);
        cv::Vec2d bottomS2next = camera.worldToPixel(bottomS2n, cameraPose);

        cv::Point bottomS1currentPoint(bottomS1current[0],bottomS1current[1]);
        cv::Point bottomS1nextPoint(bottomS1next[0],bottomS1next[1]);
        cv::Point bottomS2currentPoint(bottomS2current[0],bottomS2current[1]);
        cv::Point bottomS2nextPoint(bottomS2next[0],bottomS2next[1]);
        
        if (camera.isWorldWithinFOV(bottomS1c, cameraPose) && camera.isWorldWithinFOV(bottomS1n, cameraPose))
            cv::line(image,bottomS1currentPoint,bottomS1nextPoint, cv::Scalar(0, 255, 0), 3);
        if (camera.isWorldWithinFOV(bottomS2c, cameraPose) && camera.isWorldWithinFOV(bottomS2n, cameraPose))
            cv::line(image,bottomS2currentPoint,bottomS2nextPoint, cv::Scalar(0, 255, 0), 3);
    }

    for(int i = 0;i<4*(boardHeight-1);i++){

        cv::Vec3d bottomS1c(0,squareSize * i/4,height);
        cv::Vec3d bottomS1n(0,squareSize * (i+1)/4,height);

        cv::Vec3d bottomS2c(squareSize * (boardWidth-1),squareSize * i/4,height);
        cv::Vec3d bottomS2n(squareSize * (boardWidth-1),squareSize * (i+1)/4,height);

        cv::Vec2d bottomS1current = camera.worldToPixel(bottomS1c, cameraPose);
        cv::Vec2d bottomS1next = camera.worldToPixel(bottomS1n, cameraPose);
        cv::Vec2d bottomS2current = camera.worldToPixel(bottomS2c, cameraPose);
        cv::Vec2d bottomS2next = camera.worldToPixel(bottomS2n, cameraPose);

        cv::Point bottomS1currentPoint(bottomS1current[0],bottomS1current[1]);
        cv::Point bottomS1nextPoint(bottomS1next[0],bottomS1next[1]);
        cv::Point bottomS2currentPoint(bottomS2current[0],bottomS2current[1]);
        cv::Point bottomS2nextPoint(bottomS2next[0],bottomS2next[1]);

        if (camera.isWorldWithinFOV(bottomS1c, cameraPose) && camera.isWorldWithinFOV(bottomS1n, cameraPose))
            cv::line(image,bottomS1currentPoint,bottomS1nextPoint, cv::Scalar(255, 0, 0), 3);
        if (camera.isWorldWithinFOV(bottomS2c, cameraPose) && camera.isWorldWithinFOV(bottomS2n, cameraPose))
            cv::line(image,bottomS2currentPoint,bottomS2nextPoint, cv::Scalar(255, 0, 0), 3);
    }



    for(int i = 0;i<2;i++){
        int N = 100;
        for(int j =0;j<N;j++){
            cv::Vec3d bottomS1c(squareSize * (boardWidth-1)*i,0,j*height/N);
            cv::Vec3d bottomS1n(squareSize * (boardWidth-1)*i,0,(j+1)*height/N);

            cv::Vec3d bottomS2c(squareSize * (boardWidth-1)*i,squareSize * (boardHeight-1),j*height/N);
            cv::Vec3d bottomS2n(squareSize * (boardWidth-1)*i,squareSize * (boardHeight-1),(j+1)*height/N);

            cv::Vec2d bottomS1current = camera.worldToPixel(bottomS1c, cameraPose);
            cv::Vec2d bottomS1next = camera.worldToPixel(bottomS1n, cameraPose);
            cv::Vec2d bottomS2current = camera.worldToPixel(bottomS2c, cameraPose);
            cv::Vec2d bottomS2next = camera.worldToPixel(bottomS2n, cameraPose);

            cv::Point bottomS1currentPoint(bottomS1current[0],bottomS1current[1]);
            cv::Point bottomS1nextPoint(bottomS1next[0],bottomS1next[1]);
            cv::Point bottomS2currentPoint(bottomS2current[0],bottomS2current[1]);
            cv::Point bottomS2nextPoint(bottomS2next[0],bottomS2next[1]);

            if (camera.isWorldWithinFOV(bottomS1c, cameraPose) && camera.isWorldWithinFOV(bottomS1n, cameraPose))
                cv::line(image,bottomS1currentPoint,bottomS1nextPoint, cv::Scalar(0, 0, 255), 3);
            if (camera.isWorldWithinFOV(bottomS2c, cameraPose) && camera.isWorldWithinFOV(bottomS2n, cameraPose))
                cv::line(image,bottomS2currentPoint,bottomS2nextPoint, cv::Scalar(0, 0,255 ), 3);
        }
    }
}

void ChessboardImage::recoverPose(const Chessboard & chessboard, const Camera & camera)
{
    std::vector<cv::Point3f> rPNn_all = chessboard.gridPoints();

    cv::Mat Thetacn, rNCc;
    cv::solvePnP(rPNn_all, corners, camera.cameraMatrix, camera.distCoeffs, Thetacn, rNCc);

    cv::Mat Rcn;
    cv::Rodrigues(Thetacn, Rcn);
    cameraPose.Rnc = cv::Mat(Rcn.t());
    cameraPose.rCNn = cv::Mat(-Rcn.t()*rNCc);
}

ChessboardData::ChessboardData(const std::filesystem::path & configPath)
{
    assert(std::filesystem::exists(configPath));
    cv::FileStorage fs(configPath.string(), cv::FileStorage::READ);
    assert(fs.isOpened());
    cv::FileNode node = fs["chessboard_data"];
    node["chessboard"] >> chessboard;
    std::cout << "Chessboard: " << chessboard << std::endl;

    std::string pattern;
    node["file_regex"] >> pattern;
    fs.release();
    std::regex re(pattern, std::regex_constants::basic | std::regex_constants::icase);
    
    // Populate chessboard images from regex
    std::filesystem::path root = configPath.parent_path();
    std::cout << "Scanning directory " << root.string() << " for file pattern \"" << pattern << "\"" << std::endl;
    chessboardImages.clear();
    if (std::filesystem::exists(root) && std::filesystem::is_directory(root))
    {
        for (const auto & p : std::filesystem::recursive_directory_iterator(root))
        {
            if (std::filesystem::is_regular_file(p))
            {
                if (std::regex_match(p.path().filename().string(), re))
                {
                    std::cout << "Loading " << p.path().filename().string() << "..." << std::flush;
                    cv::Mat image = cv::imread(p.path().string());

                    if (image.empty())
                    {
                        std::cout << "Error: Failed to load the image." << std::endl;
                        continue;  // Skip this image and proceed to the next one
                    }

                    std::cout << "done, detecting chessboard..." << std::flush;
                    ChessboardImage ci(image, chessboard, p.path().filename());
                    std::cout << (ci.isFound ? "found" : "not found") << std::endl;
                    
                    if (ci.isFound)
                    {
                        chessboardImages.push_back(ci);
                    }
                }
            }
        }
    }
}

void ChessboardData::drawCorners()
{
    for (auto & chessboardImage : chessboardImages)
    {
        chessboardImage.drawCorners(chessboard);
    }
}

void ChessboardData::drawBoxes(const Camera & camera)
{
    for (auto & chessboardImage : chessboardImages)
    {
        chessboardImage.drawBox(chessboard, camera);
    }
}

void ChessboardData::recoverPoses(const Camera & camera)
{
    for (auto & chessboardImage : chessboardImages)
    {
        chessboardImage.recoverPose(chessboard, camera);
    }
}

void Camera::calibrate(ChessboardData & chessboardData)
{
    std::vector<cv::Point3f> rPNn_all = chessboardData.chessboard.gridPoints();

    std::vector<std::vector<cv::Point2f>> rQOi_all;
    for (const auto & chessboardImage : chessboardData.chessboardImages)
    {
        rQOi_all.push_back(chessboardImage.corners);
    }
    assert(!rQOi_all.empty());

    imageSize = chessboardData.chessboardImages[0].image.size();
    
    flags = cv::CALIB_RATIONAL_MODEL | cv::CALIB_THIN_PRISM_MODEL;

    // Find intrinsic and extrinsic camera parameters
    cameraMatrix = cv::Mat::eye(3, 3, CV_64F);
    distCoeffs = cv::Mat::zeros(12, 1, CV_64F);
    std::vector<cv::Mat> Thetacn_all, rNCc_all;
    double rms;
    std::cout << "Calibrating camera... " << std::flush;
    
    std::vector<std::vector<cv::Point3f>> objectPoints(chessboardData.chessboardImages.size(), rPNn_all);
    rms = cv::calibrateCamera(objectPoints, rQOi_all, imageSize, cameraMatrix, distCoeffs, Thetacn_all, rNCc_all , flags);
    rms_Cam = rms;

    std::cout << "done" << std::endl;
    
    // Calculate horizontal, vertical and diagonal field of view
    calcFieldOfView();

    // Write extrinsic parameters for each chessboard image
    assert(chessboardData.chessboardImages.size() == rNCc_all.size());
    assert(chessboardData.chessboardImages.size() == Thetacn_all.size());
    for (std::size_t i = 0; i < chessboardData.chessboardImages.size(); ++i)
    {
        cv::Mat Rcn;
        cv::Rodrigues(Thetacn_all[i], Rcn);
        Pose & cameraPose = chessboardData.chessboardImages[i].cameraPose;
        
        cameraPose.Rnc = cv::Matx33d(Rcn).t();
        cameraPose.rCNn = -cameraPose.Rnc * cv::Vec3d(rNCc_all[i]);
    }
    
    printCalibration();
    std::cout << std::setw(30) << "RMS reprojection error: " << rms << std::endl;

    assert(cv::checkRange(cameraMatrix));
    assert(cv::checkRange(distCoeffs));
}

void Camera::printCalibration() const
{
    std::bitset<8*sizeof(flags)> bitflag(flags);
    std::cout << std::endl << "Calibration data:" << std::endl;
    std::cout << std::setw(30) << "Bit flags: " << bitflag << std::endl;
    std::cout << std::setw(30) << "cameraMatrix:\n" << cameraMatrix << std::endl;
    std::cout << std::setw(30) << "distCoeffs:\n" << distCoeffs.t() << std::endl;
    std::cout << std::setw(30) << "Focal lengths: " 
              << "(fx, fy) = "
              << "("<< cameraMatrix.at<double>(0, 0) << ", "<< cameraMatrix.at<double>(1, 1) << ")"
              << std::endl;       
    std::cout << std::setw(30) << "Principal point: " 
              << "(cx, cy) = "
              << "("<< cameraMatrix.at<double>(0, 2) << ", "<< cameraMatrix.at<double>(1, 2) << ")"
              << std::endl;     
    std::cout << std::setw(30) << "Field of view (horizontal): " << 180.0/CV_PI*hFOV << " deg" << std::endl;
    std::cout << std::setw(30) << "Field of view (vertical): " << 180.0/CV_PI*vFOV << " deg" << std::endl;
    std::cout << std::setw(30) << "Field of view (diagonal): " << 180.0/CV_PI*dFOV << " deg" << std::endl;
}

void Camera::calcFieldOfView()
{
    assert(cameraMatrix.rows == 3);
    assert(cameraMatrix.cols == 3);
    assert(cameraMatrix.type() == CV_64F);

    // TODO:
    hFOV = 0.0;
    vFOV = 0.0;
    dFOV = 0.0;

    // TODO:
    // Get the center, corner, and edge points in the image
    //cv::Vec2d centerPoint(imageSize.width / 2.0, imageSize.height / 2.0);
    cv::Vec2d cornerPoint1(0, 0);
    cv::Vec2d cornerPoint2(imageSize.width,imageSize.height);
    cv::Vec2d edgePointX1(imageSize.width, imageSize.height / 2.0);
    cv::Vec2d edgePointX2(0, imageSize.height / 2.0);
    cv::Vec2d edgePointY1(imageSize.width / 2.0, imageSize.height);
    cv::Vec2d edgePointY2(imageSize.width / 2.0, 0);

    // Compute the direction vectors for these points
    //cv::Vec3d centerVector = pixelToVector(centerPoint);
    cv::Vec3d cornerVector1 = pixelToVector(cornerPoint1);
    cv::Vec3d cornerVector2 = pixelToVector(cornerPoint2);
    cv::Vec3d edgeVectorX1 = pixelToVector(edgePointX1);
    cv::Vec3d edgeVectorY1 = pixelToVector(edgePointY1);
    cv::Vec3d edgeVectorX2 = pixelToVector(edgePointX2);
    cv::Vec3d edgeVectorY2 = pixelToVector(edgePointY2);

    // Compute the dot product between the center vector and other vectors
    //double CornerCornerDotProduct = centerVector1.dot(cornerVector2);
    //double centerEdgeDotProductX = centerVector.dot(edgeVectorX);
    //double centerEdgeDotProductY = centerVector.dot(edgeVectorY);
    
    double CornerCornerDotProduct = cornerVector1.dot(cornerVector2);
    double EdgeEdgeDotProductX = edgeVectorX1.dot(edgeVectorX2);
    double EdgeEdgeDotProductY = edgeVectorY1.dot(edgeVectorY2);

    // Compute the magnitudes of the vectors
    //double centerVectorMagnitude = cv::norm(centerVector);
    //double cornerVectorMagnitude = cv::norm(cornerVector);
    //double edgeVectorMagnitudeX = cv::norm(edgeVectorX);
    //double edgeVectorMagnitudeY = cv::norm(edgeVectorY);
    double edgeVectorX1Mag = cv::norm(edgeVectorX1);
    double edgeVectorX2Mag = cv::norm(edgeVectorX2);

    double edgeVectorY1Mag = cv::norm(edgeVectorY1);
    double edgeVectorY2Mag = cv::norm(edgeVectorY2);

    
    double cornerVector1Mag = cv::norm(cornerVector1);
    double cornerVector2Mag = cv::norm(cornerVector2);


    // Compute the field of view angles using the dot product formula: dot(v1, v2) = ||v1|| ||v2|| cos(theta)
    hFOV = std::acos(EdgeEdgeDotProductX / (edgeVectorX1Mag * edgeVectorX2Mag));
    vFOV = std::acos(EdgeEdgeDotProductY / (edgeVectorY1Mag * edgeVectorY2Mag));
    dFOV = std::acos(CornerCornerDotProduct / (cornerVector1Mag * cornerVector2Mag));
}

cv::Vec3d Camera::worldToVector(const cv::Vec3d & rPNn, const Pose & pose) const
{
    cv::Vec3d uPCc;
    cv::Vec3d rPCc = pose.Rnc.t() * (rPNn - pose.rCNn);
    uPCc = rPCc / cv::norm(rPCc);
    return uPCc;
}

cv::Vec2d Camera::worldToPixel(const cv::Vec3d & rPNn, const Pose & pose) const
{
    return vectorToPixel(worldToVector(rPNn, pose));
}

Eigen::Vector2d Camera::worldToPixel(const Eigen::Vector3d & rPNn, const Eigen::Vector6d & eta, Eigen::Matrix23d & JrPNn, Eigen::Matrix26d & Jeta) const
{
    // Set elements of JrPNn
    // TODO: Lab 7
    assert(false);
    

    // Set elements of Jeta
    // TODO: Lab 7

    return worldToPixel(rPNn, eta);
    // Note: If you use autodiff, return the evaluated function value (cast with double scalar type) instead of calling worldToPixel as above
}

cv::Vec2d Camera::vectorToPixel(const cv::Vec3d & rPCc) const
{
    cv::Vec2d rQOi;
        
    cv::Vec3d uPCc = rPCc/cv::norm(rPCc);


    // Form a vector of object points from the unit vector
    std::vector<cv::Vec3d> objectPoints{uPCc};

    // Vector to hold output image points
    std::vector<cv::Vec2d> imagePoints;

    // Project the 3D points to 2D image points
    cv::projectPoints(objectPoints, cv::Vec3d::zeros(), cv::Vec3d::zeros(), cameraMatrix, distCoeffs, imagePoints);

    // Extract the first (and only) projected image point
    rQOi = imagePoints[0];
    
    return rQOi;
}

Eigen::Vector2d Camera::vectorToPixel(const Eigen::Vector3d & rPCc, Eigen::Matrix23d & J) const
{
    Eigen::Vector2d rQOi;
    // TODO: Lab 7 (optional)
/*
    Eigen::VectorX<autodiff::var> xvar = rPCc.cast<autodiff::var>();
    Eigen::VectorX<autodiff::var> fvar = vectorToPixel(xvar); // Build expression tree
    J.resize(fvar.size(), xvar.size());
    for (Eigen::Index i = 0; i < fvar.size(); ++i)
        J.row(i) = gradient(fvar(i), xvar); // Evaluate derivatives from tree
    return fvar.cast<double>(); // cast return value to double
*/
    return rQOi;
}

cv::Vec3d Camera::pixelToVector(const cv::Vec2d & rQOi) const
{
    cv::Vec3d uPCc;
     std::vector<cv::Vec2d> imagePoints{rQOi};
    
    // Vector to hold output object points
    std::vector<cv::Vec2d> objectPoints;

    // Convert the 2D image points to normalized 2D points
    cv::undistortPoints(imagePoints, objectPoints, cameraMatrix, distCoeffs);

    // Convert the 2D point to a 3D unit vector
    uPCc = cv::Vec3d(objectPoints[0][0], objectPoints[0][1], 1.0);

    // Normalize the vector
    uPCc = uPCc / cv::norm(uPCc);
    return uPCc;
}

bool Camera::isVectorWithinFOV(const cv::Vec3d & rPCc) const
{
    cv::Vec3d uPCc = rPCc/cv::norm(rPCc);
    assert(std::abs(cv::norm(uPCc) - 1.0) < 100*std::numeric_limits<double>::epsilon());
    

    double angleHorizontal = std::atan2(uPCc[0], uPCc[2]);
    double angleVertical = std::atan2(uPCc[1], uPCc[2]);
    double angleDiagonal = std::atan2(std::sqrt(uPCc[0] * uPCc[0] + uPCc[1] * uPCc[1]), uPCc[2]);


   if (std::abs(angleDiagonal) <= (dFOV)/2 && uPCc[0]!=1 && uPCc[1]!=1)
    {

        return true;
    }
    else
    {
    
        return false;
    }
}

bool Camera::isWorldWithinFOV(const cv::Vec3d & rPNn, const Pose & pose) const
{
    return isVectorWithinFOV(worldToVector(rPNn, pose));
}

void Camera::write(cv::FileStorage & fs) const
{
    fs << "{"
       << "camera_matrix"           << cameraMatrix
       << "distortion_coefficients" << distCoeffs
       << "flags"                   << flags
       << "imageSize"               << imageSize
       << "}";
}

void Camera::read(const cv::FileNode & node)
{
    node["camera_matrix"]           >> cameraMatrix;
    node["distortion_coefficients"] >> distCoeffs;
    node["flags"]                   >> flags;
    node["imageSize"]               >> imageSize;

    calcFieldOfView();

    assert(cameraMatrix.cols == 3);
    assert(cameraMatrix.rows == 3);
    assert(cameraMatrix.type() == CV_64F);
    assert(distCoeffs.cols == 1);
    assert(distCoeffs.type() == CV_64F);
}
