#include <Eigen/Core>
#include "rosenbrock.hpp"

// Functor for Rosenbrock function and its derivatives
double RosenbrockAnalytical::operator()(const Eigen::VectorXd &x)
{
    return rosenbrock(x);
}

double RosenbrockAnalytical::operator()(const Eigen::VectorXd &x, Eigen::VectorXd &g)
{
    g.resize(2, 1);
    // TODO: Write gradient to g
    // f = (1-x1)^2 + 100(x2-x1^2)^2

    //double dfdx1 = -2*(1-x(0)) - 400*x(0)*(x(1) - x(0)*x(0));
    //double dfdx2 = 200*(x(1)-x(0)*x(0));

    g<<-2*(1-x(0)) - 400*x(0)*(x(1) - x(0)*x(0)),200*(x(1)-x(0)*x(0));

    return operator()(x);
}

double RosenbrockAnalytical::operator()(const Eigen::VectorXd &x, Eigen::VectorXd &g, Eigen::MatrixXd &H)
{
    H.resize(2, 2);
    // TODO: Write Hessian to H
    //operator()(x, g);

    //double d2fdx1_2 = 2 - 400*x(1) + 1200*x(0)*x(0);
   // double d2fdx2_2 = 200;

    H<<800*x(0)*x(0)-400*(x(1)-x(0)*x(0)) + 2,-400*x(0),
        -400*x(0),200;

    return operator()(x, g);
}
