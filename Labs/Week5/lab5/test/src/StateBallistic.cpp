#include <doctest/doctest.h>
#include <cstddef>
#include <Eigen/Core>
#include "../../src/StateBallistic.h"
#include "../../src/Gaussian.h"

#ifndef CAPTURE_EIGEN
#define CAPTURE_EIGEN(x) INFO(#x " = \n", x.format(Eigen::IOFormat(Eigen::FullPrecision, 0, ", ", ";\n", "", "", "[", "]")));
#endif

SCENARIO("StateBallistic")
{
    const std::size_t n = 3;
    Gaussian density(n);
    StateBallistic state(density);

    GIVEN("A state vector")
    {
        Eigen::VectorXd x(n);
        x <<  14000,
               -450,
             0.0005;

        WHEN("Evaluating f = state.dynamics(x)")
        {
            Eigen::VectorXd f = state.dynamics(x);

            THEN("f has expected dimensions")
            {
                REQUIRE(f.rows() == n);
                REQUIRE(f.cols() == 1);

                AND_THEN("f has expected values")
                {
                    Eigen::VectorXd f_expected(n);
                    f_expected <<             -450,
                                  2.51399051183737,
                                                 0;
                    CAPTURE_EIGEN(f);
                    CAPTURE_EIGEN(f_expected);
                    CHECK(f.isApprox(f_expected));
                }
            }
        }

        WHEN("Evaluating f = state.dynamics(x, J)")
        {
            Eigen::MatrixXd J;
            Eigen::VectorXd f = state.dynamics(x, J);

            THEN("f has expected dimensions")
            {
                REQUIRE(f.rows() == n);
                REQUIRE(f.cols() == 1);

                AND_THEN("f has expected values")
                {
                    Eigen::VectorXd f_expected(n);
                    f_expected <<             -450,
                                  2.51399051183737,
                                                 0;
                    CAPTURE_EIGEN(f);
                    CAPTURE_EIGEN(f_expected);
                    CHECK(f.isApprox(f_expected));
                }
            }

            THEN("J has expected dimensions")
            {
                REQUIRE(J.rows() == n);
                REQUIRE(J.cols() == n);

                AND_THEN("J has expected values")
                {
                    Eigen::MatrixXd J_expected(n,n);
                    J_expected <<                    0,                    1,                    0,
                                  -0.00172993748945584,  -0.0547732911637217,     24647.9810236747,
                                                     0,                    0,                    0;
                    CAPTURE_EIGEN(J);
                    CAPTURE_EIGEN(J_expected);
                    CHECK(J.isApprox(J_expected));
                }
            }
        }
    }
}
