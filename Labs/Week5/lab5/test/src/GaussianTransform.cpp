#include <doctest/doctest.h>
#include <cmath>
#include <Eigen/Core>
#include "../../src/Gaussian.h"

#ifndef CAPTURE_EIGEN
#define CAPTURE_EIGEN(x) INFO(#x " = \n", x.format(Eigen::IOFormat(Eigen::FullPrecision, 0, ", ", ";\n", "", "", "[", "]")));
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846264338327950288
#endif

// Example functions to test Gaussian transforms
static Eigen::VectorXd cartesianToPolar(const Eigen::VectorXd & x, Eigen::MatrixXd & J)
{
    Eigen::VectorXd f(2);
    f(0)    = std::atan2(x(0), x(1));
    f(1)    = x.norm();

    J.resize(2,2);
    J(0,0)  =  x(1)/x.squaredNorm();
    J(0,1)  = -x(0)/x.squaredNorm();

    J(1,0)  =  x(0)/x.norm();
    J(1,1)  =  x(1)/x.norm();

    return f;
}

SCENARIO("Gaussian transform")
{
    GIVEN("A Gaussian density")
    {
        Eigen::VectorXd mux(2);
        Eigen::MatrixXd Sxx(2,2);
        mux <<                  1,
                                3;
        Sxx <<                 10,                 0,
                                0,                 1;
        Gaussian px(mux,Sxx);

        WHEN("Transforming through function")
        {
            Gaussian py = px.transform(cartesianToPolar);

            Eigen::VectorXd muy = py.mean();
            Eigen::MatrixXd Pyy = py.cov();
            REQUIRE(muy.size() == 2);
            REQUIRE(Pyy.rows() == 2);
            REQUIRE(Pyy.cols() == 2);

            THEN("Mean matches expected values")
            {
                Eigen::VectorXd muy_exp(2);
                muy_exp << 0.321750554397, 3.16227766017;
                CAPTURE_EIGEN(muy);
                CAPTURE_EIGEN(muy_exp);
                CHECK(muy.isApprox(muy_exp));
            }

            THEN("Covariance matches expected values")
            {
                Eigen::MatrixXd Pyy_exp(2, 2);
                Pyy_exp << 9.01, 9.3919646507, 9.3919646507, 10.9;
                CAPTURE_EIGEN(Pyy);
                CAPTURE_EIGEN(Pyy_exp);
                CHECK(Pyy.isApprox(Pyy_exp));
            }
        }
    }

    GIVEN("A Gaussian density")
    {
        Eigen::VectorXd mux(2);
        Eigen::MatrixXd Sxx(2,2);
        mux <<                  1,
                                1;
        Sxx <<                1.5,                 0,
                                0,               1.8;
        Gaussian px(mux,Sxx);

        WHEN("Transforming through function with additive noise")
        {
            Eigen::MatrixXd SR(2, 2);
            SR.fill(0.);
            SR(0,0)     = 1.0*M_PI/180;
            SR(1,1)     = 0.1;
            Gaussian py = px.transformWithAdditiveNoise(cartesianToPolar, Gaussian(SR));

            Eigen::VectorXd muy = py.mean();
            Eigen::MatrixXd Pyy = py.cov();
            REQUIRE(muy.size() == 2);
            REQUIRE(Pyy.rows() == 2);
            REQUIRE(Pyy.cols() == 2);

            THEN("Mean matches expected values")
            {
                Eigen::VectorXd muy_exp(2);
                muy_exp << 0.785398163397448, 1.4142135623731;
                CAPTURE_EIGEN(muy);
                CAPTURE_EIGEN(muy_exp);
                CHECK(muy.isApprox(muy_exp));
            }

            THEN("Covariance matches expected values")
            {
                Eigen::MatrixXd Pyy_exp(2, 2);
                Pyy_exp << 1.37280461741979, -0.350017856687341, -0.350017856687341, 2.755;
                CAPTURE_EIGEN(Pyy);
                CAPTURE_EIGEN(Pyy_exp);
                CHECK(Pyy.isApprox(Pyy_exp));
            }
        }
    }
}
