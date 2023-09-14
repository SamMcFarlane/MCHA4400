#include <cstddef>
#include <numeric>
#include <vector>
#include <Eigen/Core>
#include "State.h"
#include "Camera.h"
#include "StateSLAMPointLandmarks.h"
#include "StateSLAM.h"
#include "MeasurementPointBundle.h"
#include <autodiff/forward/dual.hpp>
#include <autodiff/forward/dual/eigen.hpp>


#define DERIVATIVE_ANALYTICAL 1
#define DERIVATIVE_FORWARD_AUTODIFF 2
#define DERIVATIVE_REVERSE_AUTODIFF 3

#ifndef DERIVATIVE_MODE
#define DERIVATIVE_MODE DERIVATIVE_FORWARD_AUTODIFF // Set the derivative mode here to easily switch between methods
#endif



MeasurementPointBundle::MeasurementPointBundle(double time, const Eigen::VectorXd & y, const Camera & camera)
    : Measurement(time, y)
    , camera_(camera)
{
    // SR is an upper triangular matrix such that SR.'*SR = R is the measurement noise covariance
    const Eigen::Index & ny = y.size();
    double sigma = camera.rms_Cam;
    Eigen::MatrixXd SR = sigma*sigma*Eigen::MatrixXd::Identity(ny, ny); // TODO: Assignment(s) --> This needs to be rms reprojection error
    noise_ = Gaussian(SR);

    // useQuasiNewton = false;
}

double MeasurementPointBundle::logLikelihood(const State & state, const Eigen::VectorXd & x) const
{
    const StateSLAM & stateSLAM = dynamic_cast<const StateSLAM &>(state);

    // Select visible landmarks 
    std::vector<std::size_t> idxLandmarks(stateSLAM.numberLandmarks());
    // TODO: Assignment(s) --> implement to check if in FOV
    std::iota(idxLandmarks.begin(), idxLandmarks.end(), 0); // Select all landmarks

    Eigen::MatrixXd J;

    Eigen::VectorXd h = stateSLAM.predictSLAMPoseLandmarks(x, J,camera_, idxLandmarks);

    Gaussian likelihood(h, noise_.sqrtCov());
    return likelihood.log(y_);
}

double MeasurementPointBundle::logLikelihood(const State & state, const Eigen::VectorXd & x, Eigen::VectorXd & g) const
{
 // Evaluate gradient for SR1 and Newton methods
    // TODO: Assignment(s)
    g.resize(x.size());
    g.setZero();

    Eigen::VectorX<autodiff::dual> xdual = x.cast<autodiff::dual>();
    autodiff::dual fdual;
    g = gradient(&MeasurementPointBundle::logLikelihoodImpl<autodiff::dual>, wrt(xdual), at(this, state,xdual), fdual);
    return val(fdual);

}

double MeasurementPointBundle::logLikelihood(const State & state, const Eigen::VectorXd & x, Eigen::VectorXd & g, Eigen::MatrixXd & H) const
{
    // Evaluate Hessian for Newton method
    // TODO: Assignment(s)
    H.resize(x.size(), x.size());
    H.setZero();

    Eigen::VectorX<autodiff::dual2nd> xdual = x.cast<autodiff::dual2nd>();
    autodiff::dual2nd fdual;
    H = hessian(&MeasurementPointBundle::logLikelihoodImpl<autodiff::dual2nd>, wrt(xdual), at(this, state, xdual), fdual, g);
    return val(fdual);
}

void MeasurementPointBundle::update(State & state)
{
    StateSLAM & stateSLAM = dynamic_cast<StateSLAM &>(state);

    // TODO: Assignment(s)
    // Identify landmarks with matching features (data association)
    // Remove failed landmarks from map (consecutive failures to match)
    // Identify surplus features that do not correspond to landmarks in the map
    // Initialise up to Nmax – N new landmarks from best surplus features
    
    Measurement::update(state);  // Do the actual measurement update
}

