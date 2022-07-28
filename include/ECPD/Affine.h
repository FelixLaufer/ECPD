// MIT License, Copyright (c) 2022 Felix Laufer

#ifndef _AFFINE_H_
#define _AFFINE_H_

#include "ECPD.h"

struct Affine
{
  Matrix B = Matrix::Zero(0, 0);
  Vector t = Vector::Zero(0);
};

template<>
class ECPD<Affine> : public ECPD<>
{
public:
  struct Config : public ECPD<>::Config
  {};

  static std::pair<Affine, Model> compute(const Matrix& Xarg, const Matrix& Yarg, const Config& config = Config(), const Matrix& Pcon = Matrix::Zero(0, 0))
  {
    if (Xarg.cols() != Yarg.cols())
    {
      std::cerr << "Error: input matrices have invalid dimensions!" << std::endl;
      return { Affine(), Model() };
    }

    Matrix X = Xarg;
    Matrix Y = Yarg;

    // Normalize inputs
    NormParams nP;
    if (config.normalize)
      normalize(X, Y, nP);

    // Initialization
    const unsigned int N = X.rows();
    const unsigned int M = Y.rows();
    const unsigned int D = X.cols();

    Model model = { Matrix::Identity(M, N), Y, config.sigma2, 0 };
    Affine transform = { Matrix::Identity(D, D), Vector::Zero(D) };

    if (model.sigma2 <= 0)
      model.sigma2 = (M * (X.transpose() * X).trace() + N * (Y.transpose() * Y).trace() - 2 * X.colwise().sum() * Y.colwise().sum().transpose()) / (M * N * D);

    // Check if the problem is pre-constrained
    bool prePcon = Pcon.size() != 0;
    if (prePcon && (Pcon.rows() != Eigen::Index(M) || Pcon.cols() != Eigen::Index(N)))
    {
      prePcon = false;
      std::cerr << "Warning: Pcon has invalid dimensions. Ignoring all constraints for now..." << std::endl;
    }

    if (config.verbose)
      std::cout << "Affine ECPD problem is " << (!prePcon ? "un-" : "") << "constrained." << std::endl;

    // Optimization loop
    unsigned int iter = 0;
    ScalarType ntol = config.tol + 10;
    model.L = 0;
    for (; iter < config.maxIter && ntol > config.tol && model.sigma2 > 10 * std::numeric_limits<ScalarType>::epsilon(); ++iter)
    {
      const ScalarType LLast = model.L;

      // Compute P and L for the current GMM
      prePcon ? computeP(X, model, config.omega, config.alpha2, Pcon) : computeP(X, model, config.omega);

      ntol = std::abs((model.L - LLast) / model.L);

      // Set up and pre-compute
      const Vector P1 = model.P * Vector::Ones(N);
      const Vector PT1 = model.P.transpose() * Vector::Ones(M);
      Vector Pcon1; // constrained only
      ScalarType s2a2; // constrained only
      ScalarType Np;
      Vector muX;
      Vector muY;
      Matrix XHat;
      Matrix YHat;
      Matrix B;

      if (prePcon)
      {
        Pcon1 = Pcon * Vector::Ones(N);
        const Vector PconT1 = Pcon.transpose() * Vector::Ones(M);
        s2a2 = model.sigma2 / config.alpha2;
        Np = Vector::Ones(M).transpose() * P1 + s2a2 * ScalarType(Vector::Ones(M).transpose() * Pcon1);
        muX = (X.transpose() * PT1 + s2a2 * X.transpose() * PconT1) / Np;
        muY = (Y.transpose() * P1 + s2a2 * Y.transpose() * Pcon1) / Np;
        XHat = X - Vector::Ones(N) * muX.transpose();
        YHat = Y - Vector::Ones(M) * muY.transpose();
        B = (XHat.transpose() * model.P.transpose() * YHat) * inverse(YHat.transpose() * P1.asDiagonal() * YHat);
        B += s2a2 * (XHat.transpose() * Pcon.transpose() * YHat) * inverse(YHat.transpose() * Pcon1.asDiagonal() * YHat);
      }
      else
      {
        Np = Vector::Ones(M).transpose() * P1;
        muX = X.transpose() * PT1 / Np;
        muY = Y.transpose() * P1 / Np;
        XHat = X - Vector::Ones(N) * muX.transpose();
        YHat = Y - Vector::Ones(M) * muY.transpose();
        B = (XHat.transpose() * model.P.transpose() * YHat) * inverse(YHat.transpose() * P1.asDiagonal() * YHat);
      }

      if (prePcon)
        model.sigma2 = std::abs((XHat.transpose() * PT1.asDiagonal() * XHat).trace() - (XHat.transpose() * model.P.transpose() * YHat * B.transpose()).trace() + s2a2 * (YHat.transpose() * Pcon1.asDiagonal() * YHat).trace()) / (PT1.sum() * D);
      else
        model.sigma2 = std::abs((XHat.transpose() * PT1.asDiagonal() * XHat).trace() - (XHat.transpose() * model.P.transpose() * YHat * B.transpose()).trace()) / (Np * D);

      // Affine transform
      transform.B = B;

      // Translation
      transform.t = muX - transform.B * muY;

      // Update the GMM
      model.T = Y * transform.B.transpose() + Vector::Ones(M) * transform.t.transpose();

      if (config.verbose)
        std::cout << "Iteration #" << iter + 1 << ", sigma2 = " << model.sigma2 << std::endl;
    }

    // Compute final P and L of the final GMM
    prePcon ? computeP(X, model, config.omega, config.alpha2, Pcon) : computeP(X, model, config.omega);

    // Denormalize output results
    if (config.normalize)
      denormalize(nP, model, transform);

    return { transform, model };
  }

  static std::pair<Affine, Model> compute(const Matrix& Xarg, const Matrix& Yarg, const Config& config, const std::vector<std::pair<unsigned int, unsigned int>>& constraints)
  {
    return compute(Xarg, Yarg, config, ECPD<>::constraints2Pcon(Xarg, Yarg, constraints));
  }

protected:
  static void denormalize(const NormParams& nP, Model& model, Affine& affineTransform)
  {
    ECPD<>::denormalize(nP, model);
    const ScalarType s = nP.xScale / nP.yScale;
    affineTransform.t = nP.xScale * affineTransform.t + nP.xMean - s * affineTransform.B * nP.yMean;
    affineTransform.B = s * affineTransform.B;
  }

private:
  static Matrix pseudoInverse(const Matrix& m)
  {
    if (m.cols() != m.rows())
    {
      if (m.cols() > m.rows())
        return m.transpose() * (m * m.transpose()).fullPivLu().inverse();
      else
        return (m.transpose() * m).fullPivLu().inverse() * m.transpose();
    }

    Eigen::JacobiSVD<Matrix> svd(m, Eigen::ComputeFullU | Eigen::ComputeFullV);

    Matrix ret(m.cols(), m.rows());
    ret.setZero();

    const Vector w = svd.singularValues();

    const ScalarType sum = w.sum();
    if (sum < 1e-15)
      return ret;

    Matrix diag(w.size(), w.size());
    diag.setZero();
    for (unsigned int i = 0; i < w.size(); i++)
      diag(i, i) = (fabs(w[i] / sum) < 1e-15) ? 0. : 1. / w[i];

    ret = svd.matrixV() * diag.transpose() * svd.matrixU().transpose();
    return ret;
  }

  static Matrix inverse(const Matrix& m)
  {
    if (m.size() == 1)
      return (m.array().inverse()).eval();

    Eigen::FullPivLU<Matrix> lu(m);
    if (lu.isInvertible())
      return lu.inverse();

    return pseudoInverse(m);
  }
};

#endif
