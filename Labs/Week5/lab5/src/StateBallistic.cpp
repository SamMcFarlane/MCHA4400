#include <cmath>
#include <Eigen/Core>
#include "Gaussian.h"
#include "State.h"
#include "StateBallistic.h"

const double StateBallistic::p0 = 101.325e3;            // Air pressure at sea level [Pa]
const double StateBallistic::M  = 0.0289644;            // Molar mass of dry air [kg/mol]
const double StateBallistic::R  = 8.31447;              // Gas constant [J/(mol.K)]
const double StateBallistic::L  = 0.0065;               // Temperature gradient [K/m]
const double StateBallistic::T0 = 288.15;               // Temperature at sea level [K]
const double StateBallistic::g  = 9.81;                 // Acceleration due to gravity [m/s^2]

StateBallistic::StateBallistic(const Gaussian & density)
    : State(density)
{
    // SQ_ is an upper triangular matrix such that SQ_.'*SQ_ = Q is the power spectral density of the continuous time process noise
    SQ_ << 0,     0,    0,
           0, 1e-10,    0,
           0,     0, 5e-6;
}

// Evaluate f(x) from the SDE dx = f(x)*dt + dw
Eigen::VectorXd StateBallistic::dynamics(const Eigen::VectorXd & x) const
{
    Eigen::VectorXd f(x.size());
    double v = x(1);
    double h = x(0);
    double c = x(2);
    using std::pow;
    double d = 0.5*(M*p0/R)*(1/(T0-L*h))*pow((1 - (L*h)/(T0)),(g*M/(R*L)))*v*v*c;
    f<<v, (d-g), 0;

    

    return f;
}

// Evaluate f(x) and its Jacobian J = df/fx from the SDE dx = f(x)*dt + dw
Eigen::VectorXd StateBallistic::dynamics(const Eigen::VectorXd & x, Eigen::MatrixXd & J) const
{
    Eigen::VectorXd f = dynamics(x);

    J.resize(x.size(), x.size());
    
    double v = x(1);
    double h = x(0);
    double c = x(2);
    using std::pow;
    double d_dh = -(M*c*(M*g-L*R)*p0*v*v*pow((1 - (L*h)/(T0)),(g*M/(R*L))))/(2*R*R*(L*h - T0)*(L*h - T0));
    double d_dv = (M*p0/R)*(1/(T0-L*h))*pow((1 - (L*h)/(T0)),(g*M/(R*L)))*v*c;
    double d_dc = 0.5*(M*p0/R)*(1/(T0-L*h))*pow((1 - (L*h)/(T0)),(g*M/(R*L)))*v*v;
    J << 0, 1, 0,
        d_dh, d_dv, d_dc,
        0, 0, 0;

    return f;
}
