#ifndef GAUSSIAN_H
#define GAUSSIAN_H

#include <cstddef>
#include <cassert>
#include <Eigen/Core>
#include <Eigen/QR>

class Gaussian
{
public:
    Gaussian();
    explicit Gaussian(std::size_t n);
    explicit Gaussian(const Eigen::MatrixXd & S);
    Gaussian(const Eigen::VectorXd & mu, const Eigen::MatrixXd & S);

    std::size_t size() const;
    const Eigen::VectorXd & mean() const;
    const Eigen::MatrixXd & sqrtCov() const;
    Eigen::MatrixXd cov() const;

    // Joint distribution from product of independent Gaussians
    Gaussian operator*(const Gaussian & other) const;

    // Simulate (generate samples)
    Eigen::VectorXd simulate() const;

    // Marginal distribution
    template <typename IndexType> Gaussian marginal(const IndexType & idx) const;

    // Conditional distribution
    template <typename IndexTypeA, typename IndexTypeB> Gaussian conditional(const IndexTypeA & idxA, const IndexTypeB & idxB, const Eigen::VectorXd &xB) const;

    // Affine transform
    template <typename Func> Gaussian transform(Func h) const;
    template <typename Func> Gaussian transformWithAdditiveNoise(Func h, const Gaussian & noise) const;
protected:
    Eigen::VectorXd mu_;
    Eigen::MatrixXd S_;
};

// Given joint density p(x), return marginal density p(x(idx))
template <typename IndexType>
Gaussian Gaussian::marginal(const IndexType & idx) const
{

    Gaussian out(idx.size());
 
    out.mu_ = mu_(idx);

    Eigen::MatrixXd temp = S_(Eigen::all,idx);

    Eigen::HouseholderQR<Eigen::MatrixXd> qr(temp);

    // Extract upper triangular matrix
    out.S_ = qr.matrixQR().triangularView<Eigen::Upper>();

    return out;
}

// Given joint density p(x), return conditional density p(x(idxA) | x(idxB) = xB)
template <typename IndexTypeA, typename IndexTypeB>
Gaussian Gaussian::conditional(const IndexTypeA & idxA, const IndexTypeB & idxB, const Eigen::VectorXd & xB) const
{
    // FIXME: The following implementation is in error, but it does pass some of the unit tests
    Gaussian out;
    
    Eigen::MatrixXd Sb = S_(Eigen::all,idxB);
    Eigen::MatrixXd Sa = S_(Eigen::all,idxA);

    Eigen::MatrixXd Ssig(Sa.rows(), Sa.cols() + Sb.cols());

    // Concatenate Sa and Sb horizontally
    Ssig.block(0, 0, Sb.rows(), Sb.cols()) = Sb;
    Ssig.block(0, Sb.cols(), Sa.rows(), Sa.cols()) = Sa;

    Eigen::MatrixXd Rtotal;
    Eigen::HouseholderQR<Eigen::MatrixXd> qr(Ssig);

    Rtotal = qr.matrixQR().triangularView<Eigen::Upper>();

    Eigen::MatrixXd R1 = Rtotal.block(0, 0, idxB.size(), idxB.size());
    Eigen::MatrixXd R2 = Rtotal.block(0, idxB.size(), idxB.size(), idxA.size());
    Eigen::MatrixXd R3 = Rtotal.block(idxB.size(), idxB.size(), idxA.size(), idxA.size());

    out.mu_ = mu_(idxA) + ((R1.transpose()).triangularView<Eigen::Lower>().solve<Eigen::OnTheRight>(R2.transpose()))*(xB - mu_(idxB));


    out.S_ = R3;
    return out;
}

template <typename Func>
Gaussian Gaussian::transform(Func h) const
{
    Gaussian out;
    Eigen::MatrixXd C;
    out.mu_ = h(mu_, C);
    const std::size_t & ny = out.mu_.rows();
    Eigen::MatrixXd SS = S_*C.transpose();
    Eigen::HouseholderQR<Eigen::Ref<Eigen::MatrixXd>> qr(SS);   // In-place QR decomposition
    out.S_ = SS.topRows(ny).triangularView<Eigen::Upper>();
    return out;
}

template <typename Func>
Gaussian Gaussian::transformWithAdditiveNoise(Func h, const Gaussian & noise) const
{
    assert(noise.mean().isZero());
    Gaussian out;
    Eigen::MatrixXd C;
    out.mu_ = h(mu_, C) /*+ noise.mean()*/;
    const std::size_t & nx = mu_.rows();
    const std::size_t & ny = out.mu_.rows();
    Eigen::MatrixXd SS(nx + ny, ny);
    SS << S_*C.transpose(), noise.sqrtCov();
    Eigen::HouseholderQR<Eigen::Ref<Eigen::MatrixXd>> qr(SS);   // In-place QR decomposition
    out.S_ = SS.topRows(ny).triangularView<Eigen::Upper>();
    return out;
}

#endif
