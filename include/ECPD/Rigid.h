// MIT License, Copyright (c) 2022 Felix Laufer

#ifndef _RIGID_H_
#define _RIGID_H_

#include "ECPD.h"

struct Rigid
{
  Matrix R = Matrix::Zero(0, 0);
  Vector t = Vector::Zero(0);
  ScalarType s = 1;
};

template<>
class ECPD<Rigid> : public ECPD<>
{
public:
  struct Config : public ECPD<>::Config
  {
    bool scale = true;
    bool rot = true;
  };

  static std::pair<Rigid, Model> compute(const Matrix& Xarg, const Matrix& Yarg, const Config& config = Config(), const Matrix& Pcon = Matrix::Zero(0, 0))
  {
    if (Xarg.cols() != Yarg.cols())
    {
      std::cerr << "Error: input matrices have invalid dimensions!" << std::endl;
      return { Rigid(), Model() };
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
    Rigid transform = { Matrix::Identity(D, D), Vector::Zero(D), 1 };

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
      std::cout << "Rigid ECPD problem is " << (!prePcon ? "un-" : "") << "constrained." << std::endl;

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
      Matrix A;
      Matrix XHat;
      Matrix YHat;

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
        A = XHat.transpose() * model.P.transpose() * YHat + s2a2 * (XHat.transpose() * Pcon.transpose() * YHat);
      }
      else
      {
        Np = Vector::Ones(M).transpose() * P1;
        muX = X.transpose() * PT1 / Np;
        muY = Y.transpose() * P1 / Np;
        XHat = X - Vector::Ones(N) * muX.transpose();
        YHat = Y - Vector::Ones(M) * muY.transpose();
        A = XHat.transpose() * model.P.transpose() * YHat;
      }

      // Solve SVD
      Eigen::JacobiSVD<Matrix> svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
      const Matrix U = svd.matrixU();
      const Matrix VT = svd.matrixV().transpose();
      Matrix C = Matrix::Identity(D, D);
      if (config.rot)
        C(D - 1, D - 1) = (U * VT).determinant();

      // Rotation
      transform.R = U * C * VT;

      // Scaling
      if (config.scale)
      {
        if (prePcon)
        {
          transform.s = (A.transpose() * transform.R).trace() / (YHat.transpose() * P1.asDiagonal() * YHat + s2a2 * YHat.transpose() * Pcon1.asDiagonal() * YHat).trace();
          model.sigma2 = std::abs((XHat.transpose() * PT1.asDiagonal() * XHat).trace() - transform.s * (A.transpose() * transform.R).trace() + std::pow(transform.s, 2) * s2a2 * (YHat.transpose() * Pcon1.asDiagonal() * YHat).trace()) / (PT1.sum() * D);
        }
        else
        {
          transform.s = (A.transpose() * transform.R).trace() / (YHat.transpose() * P1.asDiagonal() * YHat).trace();
          model.sigma2 = std::abs((XHat.transpose() * PT1.asDiagonal() * XHat).trace() - transform.s * (A.transpose() * transform.R).trace()) / (Np * D);
        }
      }
      else
        model.sigma2 = std::abs((XHat.transpose() * PT1.asDiagonal() * XHat).trace() + (YHat.transpose() * P1.asDiagonal() * YHat).trace() - 2 * (A.transpose() * transform.R).trace()) / (Np * D);

      // Translation
      transform.t = muX - transform.s * transform.R * muY;

      // Update the GMM
      model.T = transform.s * Y * transform.R.transpose() + Vector::Ones(M) * transform.t.transpose();

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

  static std::pair<Rigid, Model> compute(const Matrix& Xarg, const Matrix& Yarg, const Config& config, const std::vector<std::pair<unsigned int, unsigned int>>& constraints)
  {
    return compute(Xarg, Yarg, config, ECPD<>::constraints2Pcon(Xarg, Yarg, constraints));
  }

  // Start the computation from different initial pre-rotations in parallel in order to overcome potential local minima
  static std::pair<Rigid, Model> computeRobust(const Matrix& Xarg, const Matrix& Yarg, const Config& config = Config(), const Matrix& Pcon = Matrix::Zero(0, 0), const bool quick = true)
  {
    std::vector<ScalarType> baseRots = { 0, -Pi_2, Pi_2, Pi };
    if (!quick)
    {
      baseRots.push_back(-Pi_4);
      baseRots.push_back(Pi_4);
      baseRots.push_back(-3 * Pi_4);
      baseRots.push_back(3 * Pi_4);
    }

    std::vector<Matrix> Rs;
    for (const auto& rotX : baseRots)
    {
      for (const auto& rotY : baseRots)
      {
        for (const auto& rotZ : baseRots)
          Rs.emplace_back(AngleAxis(rotX, Vector3::UnitX()).toRotationMatrix() * AngleAxis(rotY, Vector3::UnitY()).toRotationMatrix() *  AngleAxis(rotZ, Vector3::UnitZ()).toRotationMatrix());
      }
    }

    std::pair<Rigid, Model> bestRes;
    ScalarType minErr = std::numeric_limits<ScalarType>::max();
    #pragma omp parallel for default(shared)
    for (int r = 0; r < Rs.size(); ++r)
    {
      const Matrix& R = Rs[r];
      const std::pair<Rigid, Model> res = compute(Xarg, Yarg * R, config, Pcon);
      ScalarType err = 0;
      for (unsigned int p = 0; p < res.second.T.rows(); ++p)
        err += (Xarg.rowwise() - res.second.T.row(p)).cwiseAbs().minCoeff();

      #pragma omp critical
      {
        if (err < minErr)
        {
          minErr = err;
          bestRes = res;
          bestRes.first.R *= R.transpose();
        }
      }
    }
       
    return bestRes;
  }

protected:
  static void denormalize(const NormParams& nP, Model& model, Rigid& transform)
  {
    ECPD<>::denormalize(nP, model);
    transform.s *= nP.xScale / nP.yScale;
    transform.t = nP.xScale * transform.t + nP.xMean - transform.s * transform.R * nP.yMean;
  }
};

#endif
