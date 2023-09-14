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


