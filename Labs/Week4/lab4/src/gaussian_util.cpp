// Tip: Only include headers needed to parse this implementation only
#include <cassert>
#include <Eigen/Core>
#include <Eigen/QR>
#include <iostream>

#include "gaussian_util.h"

// TODO: Function implementations

void pythagoreanQR(const Eigen::MatrixXd & S1, const Eigen::MatrixXd & S2, Eigen::MatrixXd & S){
    Eigen::MatrixXd combined(S1.rows()+S2.rows(),S1.cols());
    combined << S1,S2;
    Eigen::HouseholderQR<Eigen::MatrixXd> qr(combined);
    S = qr.matrixQR().triangularView<Eigen::Upper>();
}

void conditionGaussianOnMarginal(const Eigen::VectorXd & muyxjoint, const Eigen::MatrixXd & Syxjoint, const Eigen::VectorXd & y, Eigen::VectorXd & muxcond, Eigen::MatrixXd & Sxcond){
    int ny = y.size();
    int nx = muyxjoint.size() - ny;

    // Extract Elements from Syxjoint
    Eigen::MatrixXd S1 = Syxjoint.block(0, 0, ny, ny);
    Eigen::MatrixXd S2 = Syxjoint.block(0, ny, ny, nx);
    Eigen::MatrixXd S3 = Syxjoint.block(ny, ny, nx, nx);

    // Extract elements from muyxjoint
    Eigen::VectorXd mean_y = muyxjoint.head(ny);
    Eigen::VectorXd mean_x = muyxjoint.tail(nx);

    Eigen::VectorXd y_diff = y - mean_y;
    

    muxcond = mean_x + ((S1.transpose()).triangularView<Eigen::Lower>().solve<Eigen::OnTheRight>(S2.transpose()))*y_diff;

    Sxcond = S3;
}