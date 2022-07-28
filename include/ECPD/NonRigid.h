// MIT License, Copyright (c) 2022 Felix Laufe

#ifndef _NONRIGID_H_
#define _NONRIGID_H_

#include "ECPD.h"

struct NonRigid
{
  Matrix W = Matrix::Zero(0, 0);
};

template<>
class ECPD<NonRigid> : public ECPD<>
{
public:
  struct Config : public ECPD<>::Config
  {
    ScalarType beta = 2;
    ScalarType lambda = 3;
  };

  static std::pair<NonRigid, Model> compute(const Matrix& Xarg, const Matrix& Yarg, const Config& config = Config(), const Matrix& Pcon = Matrix::Zero(0, 0))
  {
    if (Xarg.cols() != Yarg.cols())
    {
      std::cerr << "Error: input matrices have invalid dimensions!" << std::endl;
      return { NonRigid(), Model() };
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
    NonRigid transform = { Matrix::Zero(M, D) };

    if (model.sigma2 <= 0)
      model.sigma2 = (M * (X.transpose() * X).trace() + N * (Y.transpose() * Y).trace() - 2 * X.colwise().sum() * Y.colwise().sum().transpose()) / (M * N * D);

    const Matrix G = gaussianKernel(Y, config.beta);

    // Check if the problem is pre-constrained
    bool prePcon = Pcon.size() != 0;
    if (prePcon && (Pcon.rows() != Eigen::Index(M) || Pcon.cols() != Eigen::Index(N)))
    {
      prePcon = false;
      std::cerr << "Warning: Pcon has invalid dimensions. Ignoring all constraints for now..." << std::endl;
    }

    if (config.verbose)
      std::cout << "Non-Rigid ECPD problem is " << (!prePcon ? "un-" : "") << "constrained." << std::endl;

    // Optimization loop
    unsigned int iter = 0;
    ScalarType ntol = config.tol + 10;
    model.L = 0;
    for (; iter < config.maxIter && ntol > config.tol && model.sigma2 > 10 * std::numeric_limits<ScalarType>::epsilon(); ++iter)
    {
      const ScalarType LLast = model.L;

      // Compute P and L for the current GMM
      prePcon ? computeP(X, model, config.omega, config.alpha2, Pcon) : computeP(X, model, config.omega);
      model.L += config.lambda / 2 * (transform.W.transpose() * G * transform.W).trace();
      ntol = std::abs((model.L - LLast) / model.L);

      // Set up and pre-compute
      const Vector P1 = model.P * Vector::Ones(N);
      const Vector PT1 = model.P.transpose() * Vector::Ones(M);
      const Matrix PX = model.P * X;

      // Solve for W
      if (prePcon)
      {
        const Matrix PconX = Pcon * X;
        const Vector Pcon1 = Pcon * Vector::Ones(N);
        const ScalarType s2a2 = model.sigma2 / config.alpha2;
        transform.W = (P1.asDiagonal() * G + s2a2 * Pcon1.asDiagonal() * G + config.lambda * model.sigma2 * Matrix::Identity(M, M)).colPivHouseholderQr().solve(PX -P1.asDiagonal() * Y + s2a2 * (PconX - Pcon1.asDiagonal() * Y));
      }
      else
        transform.W = (P1.asDiagonal() * G + config.lambda * model.sigma2 * Matrix::Identity(M, M)).colPivHouseholderQr().solve(PX - P1.asDiagonal() * Y);

      // Update the GMM
      model.T = Y + G * transform.W;

      // Update sigma2
      model.sigma2 = std::abs(((X.transpose() * PT1.asDiagonal() * X).trace() - 2 * ((model.P * X).transpose() * model.T).trace() + (model.T.transpose() * P1.asDiagonal() * model.T).trace()) / (P1.sum() * D));
 
      // ToDo: maybe implement adaption step from Matlab implementation?

      if (config.verbose)
        std::cout << "Iteration #" << iter + 1 << ", sigma2 = " << model.sigma2 << std::endl;
    }

    // Compute final P and L of the final GMM
    prePcon ? computeP(X, model, config.omega, config.alpha2, Pcon) : computeP(X, model, config.omega);

    // Denormalize output results
    if (config.normalize)
      denormalize(nP, model);

    return { transform, model };
  }

  static std::pair<NonRigid, Model> compute(const Matrix& Xarg, const Matrix& Yarg, const Config& config, const std::vector<std::pair<unsigned int, unsigned int>>& constraints)
  {
    return compute(Xarg, Yarg, config, ECPD<>::constraints2Pcon(Xarg, Yarg, constraints));
  }

protected:
  static Matrix gaussianKernel(const Matrix& Y, const ScalarType beta)
  {
    const ScalarType k = -2 * std::pow(beta, 2);
    const unsigned int M = Y.rows();
    
    Matrix ret = Matrix(M, M);
    #pragma omp parallel for default(shared) 
    for (int i = 0; i < M; ++i)
      ret.col(i) = ((Y.array() - Y.row(i).replicate(M, 1).array()).pow(2).rowwise().sum() / k).exp();

    return ret;
  }
};

#endif
