#include <cassert>

#include <limits>
#include <vector>
#include <iostream>

#include <Eigen/Core>

#include <opencv2/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/features2d.hpp>

#include "Gaussian.hpp"
#include "StateSLAM.h"
#include "Camera.h"
#include "dataAssociation.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846264338327950288
#endif

double snn(const StateSLAM & state, const std::vector<std::size_t> & idxLandmarks, const Eigen::Matrix<double, 2, Eigen::Dynamic> & Y, const Camera & camera, std::vector<int> & idx, bool enforceJointCompatibility)
{
    double nSigma = 3.0;  // Number of standard deviations for confidence region

    assert(idxLandmarks.size() <= state.numberLandmarks());
    for (const auto & k : idxLandmarks)
    {
        assert(k < state.numberLandmarks());
    }

    const std::size_t & n = idxLandmarks.size();
    assert(n > 0);
    
    assert(Y.rows() == 2);
    assert(Y.cols() > 0);
    int m = Y.cols();

    // Pre-compute terms
    Gaussian<double> featureBundleDensity = state.predictFeatureBundleDensity(camera, idxLandmarks);

    // Index
    idx.clear();
    idx.resize(n, -1);     // -1 is the sentinel index for unassociated landmarks

    std::vector<int> midx;
    midx.resize(m);
    for (int i = 0; i < m; ++i)
    {
        midx[i] = i;
    }

    // Surprisal per unassociated landmark
    double sU = std::log(camera.imageSize.width) + std::log(camera.imageSize.height);

    double s = n*sU;

    double smin = std::numeric_limits<double>::infinity();
    std::vector<int> diff;
    std::vector<int>::iterator it, ls, space;
    diff.resize(m);
    for (int j = 0; j < n; ++j)
    {
        double dsmin    = std::numeric_limits<double>::infinity();
        double scur     = s;
        
        double snext    = 0;
        bool jcnext;

        std::vector<int> idxcur;
        idxcur = idx;
        space = idxcur.begin();
        std::advance(space, j);
        ls = std::set_difference(midx.begin(), midx.end(), idxcur.begin(), space, diff.begin());

        // associate landmark j with each unassociated feature
        for (it = diff.begin(); it < ls; ++it)
        {
            int i = *it;
            
            if (!individualCompatibility(i, j, Y, featureBundleDensity, nSigma))
            {
                continue;
            }

            std::vector<int> idxnext = idxcur;
            idxnext[j] = i;

            jcnext = jointCompatibility(idxnext, sU, Y, featureBundleDensity, nSigma, snext);
            if (enforceJointCompatibility && !jcnext)
            {
                continue;
            }

            double ds = snext - scur;
            if (ds < dsmin)
            {
                idx   = idxnext;
                dsmin = ds;
                s     = snext;
            }
        }

        // landmark j unassociated
        std::vector<int> idxnext = idxcur;
        jointCompatibility(idxnext, sU, Y, featureBundleDensity, nSigma, snext); // Only compute the surprisal
        
        // Change in surprisal
        double ds = snext - scur;
        if (ds < dsmin)
        {
            idx = idxnext;
            s   = snext;
        }
    }

    if (smin < std::numeric_limits<double>::infinity())
    {
        s = smin;
    }

    return s;
}

bool individualCompatibility(const int & i, const int & j, const Eigen::Matrix<double, 2, Eigen::Dynamic> & Y, const Gaussian<double> & density, const double & nSigma)
{
    Gaussian<double> mutableDensity = density.marginal(Eigen::seqN(2*j,2));

    return mutableDensity.isWithinConfidenceRegion(Y.col(i), nSigma);
}

bool jointCompatibility(const std::vector<int> & idx, const double & sU, const Eigen::Matrix<double, 2, Eigen::Dynamic> & Y, const Gaussian<double> & density, const double & nSigma, double & surprisal)
{
    int n = idx.size();             // Number of landmarks expected to be visible

    std::vector<int> idxi;          // Indices of measured features for associated landmarks
    idxi.reserve(n);                // Reserve maximum possible size to avoid reallocation
    std::vector<int> idxyj;         // Indices of predicted feature vector for associated landmarks
    idxyj.reserve(2*n);             // Reserve maximum possible size to avoid reallocation

    for (int k = 0; k < n; ++k)
    {
        if (idx[k] >= 0)            // If feature is associated with a landmark
        {
            idxi.push_back(idx[k]);
            idxyj.insert(idxyj.end(), {2*k, 2*k + 1});
        }
    }
    assert(2*idxi.size() == idxyj.size());

    // Number of associated landmarks
    int nA = idxi.size();   

    // Number of unassociated landmarks
    int nU = n - nA;
  
    // Set surprisal and return joint compatibility
    if (nA > 0)
    {
        Eigen::Matrix<double, Eigen::Dynamic, 1> temp(2 * nA);

        for (int k = 0; k < nA; ++k)
        {
            temp.segment(2 * k, 2) = Y.col(idxi[k]);
        }

        Gaussian<double> mutableDensity = density.marginal(idxyj);
        // Surprisal for unassociated landmarks plus surprisal for associated landmarks
        surprisal = nU*sU - mutableDensity.log(temp);      // TODO: Lab 8

        // Joint compatibility
        return mutableDensity.isWithinConfidenceRegion(temp, nSigma);       // TODO: Lab 8
    }
    else
    {
        // Surprisal for unassociated landmarks only
        surprisal = nU*sU;

        // Joint compatibility with no associated landmarks is vacuously true 
        return true;
    }
}


