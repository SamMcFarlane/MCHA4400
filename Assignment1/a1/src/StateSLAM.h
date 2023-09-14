#ifndef STATESLAM_H
#define STATESLAM_H

#include <Eigen/Core>
#include <opencv2/core/mat.hpp>
#include "Gaussian.hpp"
#include "Camera.h"
#include "State.h"

/*
 * State containing body velocities, body pose and landmark states
 *
 *     [ vBNb     ]  Body translational velocity (body-fixed)
 *     [ omegaBNb ]  Body angular velocity (body-fixed)
 * x = [ rBNn     ]  Body position (world-fixed)
 *     [ Thetanb  ]  Body orientation (world-fixed)
 *     [ m        ]  Landmark map states (undefined in this class)
 *
 */
class StateSLAM : public State
{
public:
    explicit StateSLAM(const Gaussian<double> & density);
    virtual StateSLAM * clone() const = 0;
    virtual Eigen::VectorXd dynamics(const Eigen::VectorXd & x) const override;
    virtual Eigen::VectorXd dynamics(const Eigen::VectorXd & x, Eigen::MatrixXd & J) const override;

    virtual Gaussian<double> bodyPositionDensity() const;
    virtual Gaussian<double> bodyOrientationDensity() const;
    virtual Gaussian<double> bodyTranslationalVelocityDensity() const;
    virtual Gaussian<double> bodyAngularVelocityDensity() const;

    template <typename Scalar> static Eigen::Vector3<Scalar> cameraPosition(const Camera & cam, const Eigen::VectorX<Scalar> & x);
    template <typename Scalar> static Eigen::Vector3<Scalar> cameraOrientation(const Camera & cam, const Eigen::VectorX<Scalar> & x);
    static Eigen::Vector3d cameraPosition(const Camera & cam, const Eigen::VectorXd & x, Eigen::MatrixXd & J);
    static Eigen::Vector3d cameraOrientation(const Camera & cam, const Eigen::VectorXd & x, Eigen::MatrixXd & J);

    virtual Gaussian<double> cameraPositionDensity(const Camera & cam) const;
    virtual Gaussian<double> cameraOrientationDensity(const Camera & cam) const;

    virtual std::size_t numberLandmarks() const = 0;
    virtual Gaussian<double> landmarkPositionDensity(std::size_t idxLandmark) const;
    virtual std::size_t landmarkPositionIndex(std::size_t idxLandmark) const = 0;

    template <typename Scalar> Eigen::Vector2<Scalar> predictFeature(const Eigen::VectorX<Scalar> & x, const Camera & cam, std::size_t idxLandmark) const;
    Eigen::Vector2d predictFeature(const Eigen::VectorXd & x, Eigen::MatrixXd & J, const Camera & cam, std::size_t idxLandmark) const;
    Gaussian<double> predictFeatureDensity(const Camera & cam, std::size_t idxLandmark) const;
    Gaussian<double> predictFeatureDensity(const Camera & cam, std::size_t idxLandmark, const Gaussian<double> & noise) const;

    template <typename Scalar> Eigen::VectorX<Scalar> predictFeatureBundle(const Eigen::VectorX<Scalar> & x, const Camera & cam, const std::vector<std::size_t> & idxLandmarks) const;
    Eigen::VectorXd predictFeatureBundle(const Eigen::VectorXd & x, Eigen::MatrixXd & J, const Camera & cam, const std::vector<std::size_t> & idxLandmarks) const;
    Gaussian<double> predictFeatureBundleDensity(const Camera & cam, const std::vector<std::size_t> & idxLandmarks) const;
    Gaussian<double> predictFeatureBundleDensity(const Camera & cam, const std::vector<std::size_t> & idxLandmarks, const Gaussian<double> & noise) const;
    template <typename Scalar> Eigen::VectorX<Scalar> StateSLAMPoseLandmarks(const Eigen::VectorX<Scalar> & x, const Camera & cam, std::size_t idxLandmark) const;
    template <typename Scalar> Eigen::VectorX<Scalar> predictSLAMPoseLandmarks(const Eigen::VectorX<Scalar> & x,const Camera & cam, const std::vector<std::size_t> & idxLandmarks) const;
    Eigen::VectorXd StateSLAMPoseLandmarks(const Eigen::VectorXd & x, Eigen::MatrixXd & J, const Camera & cam, std::size_t idxLandmark) const;
    Eigen::VectorXd predictSLAMPoseLandmarks(const Eigen::VectorXd & x, Eigen::MatrixXd & J, const Camera & cam, const std::vector<std::size_t> & idxLandmarks) const;

    cv::Mat & view();
    const cv::Mat & view() const;

protected:
    cv::Mat view_;

};

#include "rotation.hpp"

template <typename Scalar>
Eigen::Vector3<Scalar> StateSLAM::cameraPosition(const Camera & cam, const Eigen::VectorX<Scalar> & x)
{
    Eigen::Vector3<Scalar> rBNn = x.template segment<3>(6);
    Eigen::Vector3<Scalar> Thetanb = x.template segment<3>(9);
    Eigen::Matrix3<Scalar> Rnb = rpy2rot(Thetanb); 
    Eigen::Vector3<Scalar> rCNn = rBNn + Rnb*cam.rCBb;
    return rCNn;
}

template <typename Scalar>
Eigen::Vector3<Scalar> StateSLAM::cameraOrientation(const Camera & cam, const Eigen::VectorX<Scalar> & x)
{
    Eigen::Vector3<Scalar> Thetanb = x.template segment<3>(9);
    Eigen::Matrix3<Scalar> Rnb = rpy2rot(Thetanb); 
    Eigen::Matrix3<Scalar> Rnc = Rnb*cam.Rbc;
    Eigen::Vector3<Scalar> Thetanc = rot2rpy(Rnc);
    return Thetanc;
}

// Image feature location for a given landmark
template <typename Scalar>
Eigen::Vector2<Scalar> StateSLAM::predictFeature(const Eigen::VectorX<Scalar> & x, const Camera & cam, std::size_t idxLandmark) const
{
    // Obtain body pose from state
    Eigen::Vector3<Scalar> rBNn = x.template segment<3>(6);

    Eigen::Vector3<Scalar> Thetanb = x.template segment<3>(9);

    Eigen::Matrix3<Scalar> Rnb = rpy2rot(Thetanb); 

    // Pose of camera w.r.t. body
    const Eigen::Vector3d & rCBb = cam.rCBb;
    const Eigen::Matrix3d & Rbc = cam.Rbc;

    // Obtain camera pose from body pose
    Eigen::Vector3<Scalar> rCNn;
    Eigen::Matrix3<Scalar> Rnc;

    Rnc = Rnb*Rbc;

    rCNn = rBNn + Rnb*rCBb;

    // Obtain landmark position from state
    std::size_t idx = landmarkPositionIndex(idxLandmark);
    Eigen::Vector3<Scalar> rPNn = x.template segment<3>(idx);


    // Camera vector
    Eigen::Vector3<Scalar> rPCc;


    rPCc = Rnc.transpose()*(rPNn - rCNn);

    // Pixel coordinates
    Eigen::Vector2<Scalar> rQOi;


    rQOi = cam.vectorToPixel(rPCc); 
    

    return rQOi;
}

// Image feature locations for a bundle of landmarks
template <typename Scalar>
Eigen::VectorX<Scalar> StateSLAM::predictFeatureBundle(const Eigen::VectorX<Scalar> & x, const Camera & cam, const std::vector<std::size_t> & idxLandmarks) const
{
    const std::size_t & nL = idxLandmarks.size();
    const std::size_t & nx = size();
    assert(x.size() == nx);

    Eigen::VectorX<Scalar> h(2*nL);
    for (std::size_t i = 0; i < nL; ++i)
    {
        Eigen::Vector2<Scalar> rQOi = predictFeature(x, cam, idxLandmarks[i]);
        // Set pair of elements of h
        // TODO: Lab 8
        //h(2*i) = rQOi[0];
        //h(2*i + 1) = rQOi[1];
        h.segment(2 * i, 2) = rQOi;
        //std::cout<<"STATESLAM.H:::::::::: h = "<<h<<std::endl;
    }
    return h;
}




template <typename Scalar>
Eigen::VectorX<Scalar> StateSLAM::StateSLAMPoseLandmarks(const Eigen::VectorX<Scalar> & x, const Camera & cam, std::size_t idxLandmark) const{

    // Obtain body pose from state
    Eigen::Vector3<Scalar> rBNn = x.template segment<3>(6);

    Eigen::Vector3<Scalar> Thetanb = x.template segment<3>(9);

    Eigen::Matrix3<Scalar> Rnb = rpy2rot(Thetanb); 

    // Obtain landmark pose from state

    std::size_t idx = landmarkPositionIndex(idxLandmark);

    Eigen::Vector3<Scalar> rjNn = x.template segment<3>(idx);

    Eigen::Vector3<Scalar> Thetanj = x.template segment<3>(idx+3);

    Eigen::Matrix3<Scalar> Rnj = rpy2rot(Thetanj); 

    // Set corners --> TODO(s): Make it so its not constructing every time

    Scalar l = 166e-3; //m

    Eigen::VectorX<Scalar> rJcJj;
    rJcJj<<-l/2, l/2, 0,l/2, l/2, 0,l/2, -l/2, 0,-l/2, -l/2, 0;

    Eigen::VectorX<Scalar> rJcNn;
    Eigen::VectorX<Scalar> y;
    Eigen::VectorX<Scalar> rJcCc;
    Eigen::MatrixX<Scalar> Rbc = cam.Rbc;

    y.resize(8);

    for(int i=0;i<4;i++){
        Eigen::Vector3<Scalar> rJcNn = Rnj*rJcJj.template segment<3>(i*3) + rjNn;
        rJcCc = Rbc.transpose()*(rJcNn - rBNn); 

        y.template segment<2>(i*2) = cam.vectorToPixel<Scalar>(rJcCc);
    };
    

    return y; // TODO(s): Make unit test --> Every ArUco tag is going to have a 8x1 measurement from the corners (pixel coords)

}


template <typename Scalar>
Eigen::VectorX<Scalar> StateSLAM::predictSLAMPoseLandmarks(const Eigen::VectorX<Scalar> & x,const Camera & cam, const std::vector<std::size_t> & idxLandmarks) const{
    const std::size_t & nL = idxLandmarks.size();
    const std::size_t & nx = size();
    assert(x.size() == nx);

    Eigen::VectorX<Scalar> h(8*nL);
    for (std::size_t i = 0; i < nL; ++i)
    {
        Eigen::VectorX<Scalar> rQOi = StateSLAMPoseLandmarks(x, cam, idxLandmarks[i]);
        // Set pair of elements of h
        // TODO: Lab 8
        //h(2*i) = rQOi[0];
        //h(2*i + 1) = rQOi[1];
        h.segment(8 * i, 8) = rQOi;
        //std::cout<<"STATESLAM.H:::::::::: h = "<<h<<std::endl;
    }
    return h;
}

#endif
