#include <cmath>
#include <Eigen/Core>
#include "Gaussian.h"
#include "State.h"
#include "MeasurementRADAR.h"

const double MeasurementRADAR::r1 = 5000;    // Horizontal position of sensor [m]
const double MeasurementRADAR::r2 = 5000;    // Vertical position of sensor [m]

MeasurementRADAR::MeasurementRADAR(double time, const Eigen::VectorXd & y)
    : Measurement(time, y)
{
    // SR is an upper triangular matrix such that SR.'*SR = R is the measurement noise covariance
    Eigen::MatrixXd SR(1, 1);
    SR << 50.0;
    noise_ = Gaussian(SR);
}

// Evaluate h(x) from the measurement model y = h(x) + v
Eigen::VectorXd MeasurementRADAR::predict(const Eigen::VectorXd & x) const
{
    Eigen::VectorXd h(1);
    using std::hypot;
    h << hypot(r1,(x(0)-r2));
    return h;
}

// Evaluate h(x) and its Jacobian J = dh/fx from the measurement model y = h(x) + v
Eigen::VectorXd MeasurementRADAR::predict(const Eigen::VectorXd & x, Eigen::MatrixXd & J) const
{
    Eigen::VectorXd h = predict(x);

    J.resize(h.size(), x.size());
    //using std::hypot;
    double hrng_dh = (x(0)-r2)/h(0);

    J<<hrng_dh,0,0;

    return h;
}
