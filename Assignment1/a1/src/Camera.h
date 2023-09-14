#ifndef CAMERA_H
#define CAMERA_H

#include <vector>
#include <filesystem>
#include <Eigen/Core>
#include <opencv2/core/types.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/persistence.hpp>
#include "serialisation.hpp"
#include <iostream>

struct Pose
{
    cv::Matx33d Rnc;
    cv::Vec3d rCNn;
};

struct Chessboard
{
    cv::Size boardSize;
    float squareSize;

    void write(cv::FileStorage & fs) const;                 // OpenCV serialisation
    void read(const cv::FileNode & node);                   // OpenCV serialisation

    std::vector<cv::Point3f> gridPoints() const;
    friend std::ostream & operator<<(std::ostream &, const Chessboard &);
};

struct Camera;

struct ChessboardImage
{
    ChessboardImage(const cv::Mat &, const Chessboard &, const std::filesystem::path & = "");
    cv::Mat image;
    std::filesystem::path filename;
    Pose cameraPose;                                        // Extrinsic camera parameters
    std::vector<cv::Point2f> corners;                       // Chessboard corners in image [rQOi]
    bool isFound;
    void drawCorners(const Chessboard &);
    void drawBox(const Chessboard &, const Camera &);
    void recoverPose(const Chessboard &, const Camera &);
};

struct ChessboardData
{
    explicit ChessboardData(const std::filesystem::path &); // Load from config file

    Chessboard chessboard;
    std::vector<ChessboardImage> chessboardImages;

    void drawCorners();
    void drawBoxes(const Camera &);
    void recoverPoses(const Camera &);
};

namespace Eigen {
using Matrix23d = Eigen::Matrix<double, 2, 3>;
using Matrix26d = Eigen::Matrix<double, 2, 6>;
using Vector6d = Eigen::Matrix<double, 6, 1>;
template <typename Scalar> using Vector6 = Eigen::Matrix<Scalar, 6, 1>;
}

struct Camera
{
    void calibrate(ChessboardData &);                       // Calibrate camera from chessboard data
    void printCalibration() const;

    cv::Vec3d worldToVector(const cv::Vec3d & rPNn, const Pose & pose) const;
    template <typename Scalar> Eigen::Vector3<Scalar> worldToVector(const Eigen::Vector3<Scalar> & rPNn, const Eigen::Vector6<Scalar> & eta) const;

    cv::Vec2d worldToPixel(const cv::Vec3d &, const Pose &) const;
    template <typename Scalar> Eigen::Vector2<Scalar> worldToPixel(const Eigen::Vector3<Scalar> & rPNn, const Eigen::Vector6<Scalar> & eta) const;
    Eigen::Vector2d worldToPixel(const Eigen::Vector3d & rPNn, const Eigen::Vector6d & eta, Eigen::Matrix23d & JrPNn, Eigen::Matrix26d & Jeta) const;

    cv::Vec2d vectorToPixel(const cv::Vec3d &) const;
    template <typename Scalar> Eigen::Vector2<Scalar> vectorToPixel(const Eigen::Vector3<Scalar> &) const;
    Eigen::Vector2d vectorToPixel(const Eigen::Vector3d &, Eigen::Matrix23d &) const;

    cv::Vec3d pixelToVector(const cv::Vec2d &) const;

    bool isWorldWithinFOV(const cv::Vec3d & rPNn, const Pose & pose) const;
    bool isVectorWithinFOV(const cv::Vec3d & rPCc) const;

    void calcFieldOfView();
    void write(cv::FileStorage &) const;                    // OpenCV serialisation
    void read(const cv::FileNode &);                        // OpenCV serialisation

    cv::Mat cameraMatrix;                                   // Camera matrix
    cv::Mat distCoeffs;                                     // Lens distortion coefficients
    int flags = 0;                                          // Calibration flags
    cv::Size imageSize;                                     // Image size
    double hFOV = 0.0;                                      // Horizonal field of view
    double vFOV = 0.0;                                      // Vertical field of view
    double dFOV = 0.0;                                      // Diagonal field of view
    double rms_Cam = 1.0;

    Eigen::Vector3d rCBb = Eigen::Vector3d::Zero();       // TODO: Assignment(s)
    Eigen::Matrix3d Rbc = Eigen::Matrix3d::Identity();    // TODO: Assignment(s)
    Camera(){
        Rbc(2,0) = 1.0;
        Rbc(0,1) = 1.0;
        Rbc(1,2) = 1.0;
    }
};

template <typename Scalar>
Eigen::Vector3<Scalar> Camera::worldToVector(const Eigen::Vector3<Scalar> & rPNn, const Eigen::Vector6<Scalar> & eta) const
{
    Eigen::Vector3<Scalar> rCNn = eta.template head<3>();
    Eigen::Vector3<Scalar> Thetanc = eta.template tail<3>();
    Eigen::Matrix3<Scalar> Rnc = rpy2rot(Thetanc);
    Eigen::Vector3<Scalar> rPCc = Rnc.transpose()*(rPNn - rCNn);
    return rPCc;
}

template <typename Scalar>
Eigen::Vector2<Scalar> Camera::worldToPixel(const Eigen::Vector3<Scalar> & rPNn, const Eigen::Vector6<Scalar> & eta) const
{
    return vectorToPixel(worldToVector(rPNn, eta));
}

#include <cmath>
#include <opencv2/calib3d.hpp>

template <typename Scalar>
Eigen::Vector2<Scalar> Camera::vectorToPixel(const Eigen::Vector3<Scalar> & rPCc) const
{
    bool isRationalModel    = (flags & cv::CALIB_RATIONAL_MODEL) == cv::CALIB_RATIONAL_MODEL;
    bool isThinPrismModel   = (flags & cv::CALIB_THIN_PRISM_MODEL) == cv::CALIB_THIN_PRISM_MODEL;
    assert(isRationalModel && isThinPrismModel);
    

    Eigen::Vector2<Scalar> rQOi;
    Scalar k1 = distCoeffs.at<double>(0);
    Scalar k2 = distCoeffs.at<double>(1);
    Scalar p1 = distCoeffs.at<double>(2);
    Scalar p2 = distCoeffs.at<double>(3);
    Scalar k3 = distCoeffs.at<double>(4);
    Scalar k4 = distCoeffs.at<double>(5);
    Scalar k5 = distCoeffs.at<double>(6);
    Scalar k6 = distCoeffs.at<double>(7);
    Scalar s1 = distCoeffs.at<double>(8);
    Scalar s2 = distCoeffs.at<double>(9);
    Scalar s3 = distCoeffs.at<double>(10);
    Scalar s4 = distCoeffs.at<double>(11);

    Scalar u = rPCc[0]/rPCc[2];
    Scalar v = rPCc[1]/rPCc[2];

    Scalar r_2 = u*u + v*v;

    Scalar alpha = k1*r_2 + k2*r_2*r_2 + k3*r_2*r_2*r_2;
    Scalar beta = k4*r_2 + k5*r_2*r_2 + k6*r_2*r_2*r_2;

    Scalar c = (1 + alpha) / (1 + beta);

    Scalar u_dot = c*u + 2*p1*u*v + p2*(r_2 + 2*u*u) + s1*r_2 + s2*r_2*r_2;
    Scalar v_dot = c*v + p1*(r_2 + 2*v*v) + 2*p2*u*v + s3*r_2 + s4*r_2*r_2;

    Scalar fx = cameraMatrix.at<double>(0, 0);
    Scalar fy = cameraMatrix.at<double>(1, 1);

    Scalar cx = cameraMatrix.at<double>(0, 2);
    Scalar cy = cameraMatrix.at<double>(1, 2);
    

    rQOi[0] = fx*u_dot + cx;
    rQOi[1] = fy*v_dot + cy;

    return rQOi;
}

#endif

