// MIT License, Copyright (c) 2022 Felix Laufer

// Coherent Point Drift: https://www.researchgate.net/publication/47544556_Point_Set_Registration_Coherent_Point_Drift
// Extended Coherent Point Drift: https://www.researchgate.net/publication/297497171_Extended_Coherent_Point_Drift_Algorithm_with_Correspondence_Priors_and_Optimal_Subsampling

#ifndef _ECPD_H_
#define _ECPD_H_

#include "Constants.h"
#include "EigenTypes.h"

#include <iostream>
#include <vector>
#include <utility>

class BaseTransform;
template<typename TTransform = BaseTransform>
class ECPD
{
public:
  struct Config
  {
    bool verbose = false;
    bool normalize = true;
    unsigned int maxIter = 150;
    ScalarType omega = 0.1;
    ScalarType sigma2 = 0;
    ScalarType tol = 1e-5;
    ScalarType alpha2 = 1;
  };

  struct Model
  {
    Matrix P = Matrix::Zero(0, 0);
    Matrix T = Matrix::Zero(0, 0);
    ScalarType sigma2 = std::numeric_limits<ScalarType>::max();
    ScalarType L = 0;
  };

  static std::pair<TTransform, Model> compute(const Matrix& Xarg, const Matrix& Yarg, const Config& config = Config(), const Matrix& Pcon = Matrix::Zero(0, 0))
  {
    return { TTransform(), Model() };
  }

  static std::pair<TTransform, Model> compute(const Matrix& Xarg, const Matrix& Yarg, const Config& config, const std::vector<std::pair<unsigned int, unsigned int>>& constraints)
  {
    return compute(Xarg, Yarg, config, constraints2Pcon(Xarg.rows(), Yarg.rows(), constraints));
  }

  static Matrix correspondences(const Matrix& P)
  {
    Matrix ret = Matrix::Zero(P.cols(), 2);
    for (unsigned int m = 0; m < P.rows(); ++m)
    {
      Eigen::Index maxIdx;
      P.row(m).maxCoeff(&maxIdx);
      ret(m, 0) = m + 1;
      ret(m, 1) = maxIdx + 1;
    }
    return ret;
  }

protected:
  struct NormParams
  {
    Vector xMean = Vector::Zero(0);
    Vector yMean = Vector::Zero(0);
    ScalarType xScale = 1;
    ScalarType yScale = 1;
  };

  static Matrix constraints2Pcon(const Matrix& X, const Matrix& Y, const std::vector<std::pair<unsigned int, unsigned int>> constraints)
  {
    if (constraints.empty())
      return Matrix::Zero(0, 0);

    const unsigned int N = X.rows();
    const unsigned int M = Y.rows();

    Matrix ret = Matrix::Zero(M, N);
    for (const auto& constraint : constraints)
    {
      if (constraint.first < N && constraint.second < M)
        ret(constraint.second, constraint.first) = 1;
      else
        std::cerr << "Warning: input constraint { " << constraint.first << ", " << constraint.second << " } is out of range. Constraint is ignored..." << std::endl;
    }

    return ret;
  }

  static void computeP(const Matrix& X, Model& model, const ScalarType omega, const ScalarType alpha2 = 1, const Matrix& Pcon = Matrix::Zero(0, 0))
  {
    const unsigned int N = X.rows();
    const unsigned int M = model.T.rows();
    const unsigned int D = X.cols();

    const ScalarType kSigma2 = -2 * model.sigma2;
    const ScalarType outlier = (omega * M * pow(-kSigma2 * Pi, 0.5 * D)) / ((1 - omega) * N);

    ScalarType L = 0;
    #pragma omp parallel for default(shared) reduction(+:L)
    for (int n = 0; n < N; ++n)
    {
      Vector nums(M);
      ScalarType denom = 0;
      for (unsigned int m = 0; m < M; ++m)
      {
        const ScalarType sNorm = (X.row(n) - model.T.row(m)).squaredNorm();
        const ScalarType num = exp(sNorm / kSigma2);
        nums(m) = num;
        denom += num;
      }

      denom += outlier;
      model.P.col(n) = nums / denom;

      L += -log(denom);
    }

    const ScalarType MConst = Pcon.size() != 0 && Pcon.rows() == model.P.rows() && Pcon.cols() == model.P.cols() ? Pcon.colwise().sum().sum() : 0;
    L += D * N * log(model.sigma2) / 2 + D * MConst * log(alpha2) / 2;

    model.L = L;
  }

  // Note: for original Myronenko Matlab implementation, set linkedScale = false
  static void normalize(Matrix& X, Matrix& Y, NormParams& nP, const bool linkedScale = false)
  {
    // Columnwise mean
    const Vector xMean = X.colwise().mean();
    const Vector yMean = Y.colwise().mean();

    // Substracting the means
    X.rowwise() -= xMean.transpose();
    Y.rowwise() -= yMean.transpose();

    // Compute scaling factors
    ScalarType xScale = std::sqrt(X.array().pow(2).sum() / X.rows());
    ScalarType yScale = std::sqrt(Y.array().pow(2).sum() / Y.rows());

    // Optionally link the two scaling factors
    if (linkedScale)
    {
      const ScalarType scale = std::max(xScale, yScale);
      xScale = scale;
      yScale = scale;
    }

    // Apply scalings
    X.array() /= xScale;
    Y.array() /= yScale;

    // Store normalization paameters
    nP = NormParams{ xMean , yMean , xScale, yScale };
  }

  static void denormalize(const NormParams& nP, Model& model)
  {
    model.T = model.T * nP.xScale + nP.xMean.transpose().replicate(model.T.rows(), 1);
  }
};

#endif
