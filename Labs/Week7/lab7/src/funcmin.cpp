#include <cstddef>
#include <cmath>
#include <limits>
#include <cassert>
#include <vector>
#include <Eigen/Core>
#include <Eigen/Eigenvalues> 
#include "funcmin.hpp"

//
// Solve trust-region subproblem
//   minimise 0.5*p.'*H*p + g.'*p
//   subject to ||p|| <= D
//
// References
// [1] Moré, J.J. and D.C. Sorensen, Computing a Trust Region Step,
//     SIAM Journal on Scientific and Statistical Computing, Vol. 3, pp
//     553--572, 1983.

int funcmin::trs(const Eigen::MatrixXd & H, const Eigen::VectorXd & g, double D, Eigen::VectorXd & p)
{
    assert(g.cols() == 1);
    assert(H.rows() == H.cols());
    assert(H.rows() == g.rows());

    p.resize(g.rows(),1);

    typedef Eigen::VectorXd Vector;
    typedef Eigen::MatrixXd Matrix;

    Eigen::SelfAdjointEigenSolver<Matrix> eigenH(H);
    const Vector & v = eigenH.eigenvalues();
    const Matrix & Q = eigenH.eigenvectors();

    return trs(Q, v, g, D, p);
}

int funcmin::trs(const Eigen::MatrixXd & Q, const Eigen::VectorXd & v, const Eigen::VectorXd & g, double D, Eigen::VectorXd & p)
{
    assert(g.cols() == 1);
    assert(v.cols() == 1);
    assert(Q.rows() == Q.cols());
    assert(Q.rows() == g.rows());
    assert(v.rows() == g.rows());

    p.resize(g.rows(),1);

    typedef double Scalar;
    typedef Eigen::VectorXd Vector;
    typedef Eigen::MatrixXd Matrix;
    typedef Vector::Index Index;

    const Scalar eps = std::numeric_limits<Scalar>::epsilon();
    const Scalar sqrteps = std::sqrt(eps);
    const int maxIterations = 20;

    Scalar l1 = v.minCoeff(); // Leftmost eigenvalue
    Vector a = Q.transpose()*g;
    
    Scalar lam;
    if (l1 < 0)
        lam = 1.01*std::fabs(l1);
    else
        lam = 0;

    Vector vlam = v.array() + lam;
    p = -Q*a.cwiseQuotient(vlam);

    if (l1 < 0 || p.norm() > D || std::fabs(lam*(p.norm() - D)) > sqrteps)
    {
        bool isHardCase = std::fabs(a(0)) < eps && l1 < 0;
        if (isHardCase)
        {
            std::vector<Index> idxValid;
            for (Index i = 0; i < v.size(); ++i)
                if (std::fabs(v(i) - l1) > sqrteps)
                    idxValid.push_back(i);

            Vector scaledValid(idxValid.size());
            Matrix QValid(v.size(),idxValid.size());
            for (std::size_t i = 0; i < idxValid.size(); ++i)
            {
                Index idx = idxValid[i];
                scaledValid(i) = a(idx)/(v(idx) - l1);
                QValid.col(i) = Q.col(idx);
            }
            Scalar t = std::sqrt(D*D - scaledValid.squaredNorm());
            Vector pvec(v.size());
            if (idxValid.size() > 0)
                pvec = QValid*scaledValid;
            else
                pvec.setZero();

            p = t*Q.col(0) - pvec;

            // Choose sign of t to give a reduction in cost
            Vector q = Q.transpose()*p;
            if (p.dot(g) + 0.5*q.transpose()*v.asDiagonal()*q > 0.0)
                p = -t*Q.col(0) - pvec;
        }
        else
        {
            int k;
            for (k = 0; k < maxIterations; ++k)
            {
                Vector pp = -a.cwiseQuotient(vlam);
                Vector dp =  a.cwiseQuotient(vlam.cwiseAbs2());
                Scalar ppnorm = pp.norm();
                Scalar ff = 1/D - 1/ppnorm;
                Scalar gg = dp.dot(pp)/(ppnorm*ppnorm*ppnorm);

                // Ensure lam > 0 and lam > l1
                lam = std::max(std::max(0.0, -l1) + sqrteps*std::max(0.0, -l1), lam - ff/gg);

                vlam = v.array() + lam;

                if (std::fabs(ff) < sqrteps)
                    break;
            }

            p = -Q*a.cwiseQuotient(vlam);
            if (k >= maxIterations)
                return 1;
        }
    }

    return 0;
}
