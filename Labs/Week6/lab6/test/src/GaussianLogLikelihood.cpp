#include <doctest/doctest.h>
#include <cmath>
#include <Eigen/Core>
#include <autodiff/forward/dual.hpp>
#include <autodiff/forward/dual/eigen.hpp>
#include "../../src/Gaussian.hpp"

#ifndef CAPTURE_EIGEN
#define CAPTURE_EIGEN(x) INFO(#x " = \n", x.format(Eigen::IOFormat(Eigen::FullPrecision, 0, ", ", ";\n", "", "", "[", "]")));
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846264338327950288
#endif

SCENARIO("Gaussian log nominal")
{
    GIVEN("Nominal parameters")
    {
        Eigen::VectorXd x(3);
        x << 1, 2, 3;
        Eigen::VectorXd mu(3);
        mu << 2, 4, 6;
        Eigen::MatrixXd S(3, 3);
        S << 1, 4, 6,
             0, 2, 5,
             0, 0, 3;
        REQUIRE_MESSAGE(S.isUpperTriangular(), "S must be upper triangular");

        Gaussian p(mu, S);
        
        WHEN("Evaluating l = p.log(x)")
        {
            double l = p.log(x);
            THEN("l matches expected value")
            {
                CHECK(l == doctest::Approx(-5.7707972911));
            }
        }

        WHEN("Evaluating l = p.log(x, g)")
        {
            Eigen::VectorXd g;
            double l = p.log(x, g);

            THEN("l matches expected value")
            {
                CHECK(l == doctest::Approx(-5.7707972911));
            }

            AND_WHEN("Computing autodiff gradient")
            {
                Eigen::VectorXd g_exp;
                Gaussian<autodiff::dual> pdual = p;
                Eigen::VectorX<autodiff::dual> xdual = x.cast<autodiff::dual>();
                autodiff::dual fdual;
                auto func = [&](const Eigen::VectorX<autodiff::dual> & x) { return pdual.log(x); };
                g_exp = gradient(func, wrt(xdual), at(xdual), fdual);
                REQUIRE(g_exp.rows() == x.size());
                REQUIRE(g_exp.cols() == 1);

                THEN("l matches autodiff value")
                {
                    CHECK(l == doctest::Approx(val(fdual)));
                }

                THEN("g has expected dimensions")
                {
                    REQUIRE(g.rows() == x.size());
                    REQUIRE(g.cols() == 1);

                    AND_THEN("g matches autodiff gradient")
                    {
                        CAPTURE_EIGEN(g);
                        CAPTURE_EIGEN(g_exp);
                        CHECK(g.isApprox(g_exp));
                    }
                }
            }
        }

        WHEN("Evaluating l = p.log(x, g, H)")
        {
            Eigen::VectorXd g;
            Eigen::MatrixXd H;
            double l = p.log(x, g, H);

            THEN("l matches expected value")
            {
                CHECK(l == doctest::Approx(-5.7707972911));
            }

            AND_WHEN("Computing autodiff gradient and Hessian")
            {
                Eigen::VectorXd g_exp;
                Eigen::MatrixXd H_exp;
                Gaussian<autodiff::dual2nd> pdual = p;
                Eigen::VectorX<autodiff::dual2nd> xdual = x.cast<autodiff::dual2nd>();
                autodiff::dual2nd fdual;
                auto func = [&](const Eigen::VectorX<autodiff::dual2nd> & x) { return pdual.log(x); };
                H_exp = hessian(func, wrt(xdual), at(xdual), fdual, g_exp);
                REQUIRE(g_exp.rows() == x.size());
                REQUIRE(g_exp.cols() == 1);
                REQUIRE(H_exp.rows() == x.size());
                REQUIRE(H_exp.cols() == x.size());

                THEN("l matches autodiff value")
                {
                    CHECK(l == doctest::Approx(val(fdual)));
                }

                THEN("g has expected dimensions")
                {
                    REQUIRE(g.rows() == x.size());
                    REQUIRE(g.cols() == 1);

                    AND_THEN("g matches autodiff gradient")
                    {
                        CAPTURE_EIGEN(g);
                        CAPTURE_EIGEN(g_exp);
                        CHECK(g.isApprox(g_exp));
                    }
                }

                THEN("H has expected dimensions")
                {
                    REQUIRE(H.rows() == x.size());
                    REQUIRE(H.cols() == x.size());

                    AND_THEN("H matches autodiff Hessian")
                    {
                        CAPTURE_EIGEN(H);
                        CAPTURE_EIGEN(H_exp);
                        CHECK(H.isApprox(H_exp));
                    }
                }
            }
        }
    }
}

SCENARIO("Gaussian log exponential underflow")
{
    GIVEN("Parameters that may cause underflow in the exponential")
    {
        Eigen::VectorXd x(1);
        x << 0;
        Eigen::VectorXd mu(1);
        mu << std::sqrt(350*std::log(10)/M_PI); // Approx 16
        Eigen::MatrixXd S(1, 1);
        S << 1.0/std::sqrt(2*M_PI); // Approx 0.4
        REQUIRE( std::exp( -0.5*S.triangularView<Eigen::Upper>().transpose().solve(x - mu).squaredNorm() ) == 0.0);

        Gaussian p(mu, S);
        
        WHEN("Evaluating l = p.log(x)")
        {
            double l = p.log(x);
            THEN("l matches expected value")
            {
                CHECK(l == doctest::Approx(-805.904782547916));
            }
        }
    }
}

#include <Eigen/LU> // For .determinant()

SCENARIO("Gaussian log determinant underflow")
{
    GIVEN("Parameters that may cause determinant underflow")
    {
        double a = 1e-4;    // Magnitude of st.dev.
        int n = 100;        // Dimension
        Eigen::VectorXd x = Eigen::VectorXd::Zero(n);
        Eigen::VectorXd mu = Eigen::VectorXd::Zero(n);
        Eigen::MatrixXd S = a*Eigen::MatrixXd::Identity(n, n);
        REQUIRE(S.determinant() == 0.0); // underflow to zero

        Gaussian p(mu, S);
        
        WHEN("Evaluating l = p.log(x)")
        {
            double l = p.log(x);
            THEN("l matches expected value")
            {
                CHECK(l == doctest::Approx(-n*std::log(a) - n/2.0*std::log(2*M_PI)));
            }
        }
    }
}

SCENARIO("Gaussian log determinant overflow")
{
    GIVEN("Parameters that may cause determinant overflow")
    {
        double a = 1e4;     // Magnitude of st.dev.
        int n = 100;        // Dimension
        Eigen::VectorXd x = Eigen::VectorXd::Zero(n);
        Eigen::VectorXd mu = Eigen::VectorXd::Zero(n);
        Eigen::MatrixXd S = a*Eigen::MatrixXd::Identity(n,n);
        REQUIRE(S.determinant() == std::numeric_limits<double>::infinity()); // overflow to infinity
        
        Gaussian p(mu,S);

        WHEN("Evaluating l = p.log(x)")
        {
            double l = p.log(x);
            THEN("l matches expected value")
            {
                CHECK(l == doctest::Approx(-n*std::log(a) - n/2.0*std::log(2*M_PI)));
            }
        }
    }
}
