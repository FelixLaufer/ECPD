// MIT license, Copyright (C) 2022 Felix Laufer

#ifndef _EIGEN_TYPES_H_
#define _EIGEN_TYPES_H_

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <complex>

// Scalar type: double / float
using ScalarType = double;
using ScalarTypeC = std::complex<ScalarType>;

// Smaller dim static size vectors
using Vector2 = Eigen::Matrix<ScalarType, 2, 1>;
using Vector3 = Eigen::Matrix<ScalarType, 3, 1>;
using Vector4 = Eigen::Matrix<ScalarType, 4, 1>;
using Vector5 = Eigen::Matrix<ScalarType, 5, 1>;
using Vector6 = Eigen::Matrix<ScalarType, 6, 1>;
using Vector7 = Eigen::Matrix<ScalarType, 7, 1>;
using Vector8 = Eigen::Matrix<ScalarType, 8, 1>;
using Vector9 = Eigen::Matrix<ScalarType, 9, 1>;

// Higher dim static size vectors
template <size_t N>
using VectorN = Eigen::Matrix<ScalarType, N, 1>;
template <size_t N>
using VectorCN = Eigen::Matrix<ScalarTypeC, N, 1>;

// Dynamic size vectors
using Vector = Eigen::Matrix<ScalarType, Eigen::Dynamic, 1>;
using VectorC = Eigen::Matrix<ScalarTypeC, Eigen::Dynamic, 1>;

// Smaller dim static size matrices
using Matrix2x2 = Eigen::Matrix<ScalarType, 2, 2>;
using Matrix3x3 = Eigen::Matrix<ScalarType, 3, 3>;
using Matrix2x3 = Eigen::Matrix<ScalarType, 2, 3>;
using Matrix3x2 = Eigen::Matrix<ScalarType, 3, 2>;
using Matrix4x4 = Eigen::Matrix<ScalarType, 4, 4>;
using Matrix4x3 = Eigen::Matrix<ScalarType, 4, 3>;
using Matrix3x4 = Eigen::Matrix<ScalarType, 3, 4>;
using Matrix4x2 = Eigen::Matrix<ScalarType, 4, 2>;
using Matrix2x4 = Eigen::Matrix<ScalarType, 2, 4>;

// Higher dim static size matrices
template <size_t N, size_t M>
using MatrixNxM = Eigen::Matrix<ScalarType, N, M>;
template <size_t N, size_t M>
using MatrixCNxM = Eigen::Matrix<ScalarTypeC, N, M>;

// Dynamic size matrices
using Matrix = Eigen::Matrix<ScalarType, Eigen::Dynamic, Eigen::Dynamic>;
using MatrixC = Eigen::Matrix<ScalarTypeC, Eigen::Dynamic, Eigen::Dynamic>;

#endif
