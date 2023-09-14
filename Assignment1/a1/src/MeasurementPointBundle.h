#ifndef MEASUREMENTPOINTBUNDLE_H
#define MEASUREMENTPOINTBUNDLE_H

#include <Eigen/Core>
#include "State.h"
#include "Camera.h"
#include "Measurement.h"
#include "StateSLAMPointLandmarks.h"
#include "StateSLAM.h"
#include <autodiff/forward/dual.hpp>
#include <autodiff/forward/dual/eigen.hpp>

class MeasurementPointBundle : public Measurement
{
public:
    MeasurementPointBundle(double time, const Eigen::VectorXd & y, const Camera & camera);
    virtual double logLikelihood(const State & state, const Eigen::VectorXd & x) const override;
    virtual double logLikelihood(const State & state, const Eigen::VectorXd & x, Eigen::VectorXd & g) const override;
    virtual double logLikelihood(const State & state, const Eigen::VectorXd & x, Eigen::VectorXd & g, Eigen::MatrixXd & H) const override;
    template <typename Scalar> Scalar logLikelihoodImpl(const State & state, const Eigen::VectorX<Scalar> & x) const;

protected:
    virtual void update(State & state) override;
    const Camera & camera_;
};


template <typename Scalar>
Scalar MeasurementPointBundle::logLikelihoodImpl(const State & state, const Eigen::VectorX<Scalar> & x) const
{

    const StateSLAM & stateSLAM = dynamic_cast<const StateSLAM &>(state);

    // Select visible landmarks 
    std::vector<std::size_t> idxLandmarks(stateSLAM.numberLandmarks());
    // TODO: Assignment(s) --> implement to check if in FOV
    std::iota(idxLandmarks.begin(), idxLandmarks.end(), 0); // Select all landmarks

    Eigen::MatrixX<Scalar> J;

    Eigen::VectorX<Scalar> h = stateSLAM.predictSLAMPoseLandmarks<Scalar>(x,camera_, idxLandmarks);

    Eigen::VectorX<Scalar> y = y_.cast<Scalar>();

    Eigen::MatrixX<Scalar> noise = noise_.sqrtCov().cast<Scalar>();

    Gaussian<Scalar> likelihood(h, noise);
    return likelihood.log(y);

}


#endif