// MIT License, Copyright (c) 2022 Felix Laufer

#include <ECPD/ECPD.h>
#include <ECPD/Rigid.h>
#include <ECPD/Affine.h>
#include <ECPD/NonRigid.h>

#include <Eigen2Mat/Matlab.h>

#include <algorithm>
#include <fstream>

Matrix parseXYZData(const std::string& file)
{
  std::fstream infile(file);
  std::vector<ScalarType> data;
  ScalarType x, y, z;
  size_t rows = 0;
  while (infile >> x >> y >> z && ++rows)
    data.insert(data.end(), { x, y, z });
  return Eigen::Map<Matrix>(data.data(), 3, rows).transpose();
}

Matrix permutateRows(const Matrix& m)
{
  Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> perm(m.rows());
  perm.setIdentity();
  std::random_shuffle(perm.indices().data(), perm.indices().data() + perm.indices().size());
  return perm * m;
}

int main(int argc, char* argv[])
{
  Matlab::Session mat;

  const auto showPCs = [&](const Matrix& X, const Matrix&Y, const bool holdOn = false)
  {
    if (X.cols() != 3 || Y.cols() != 3 || X.rows() != Y.rows())
      return;

    mat.set("X", X);
    mat.set("Y", Y);
    mat.eval("scatter3(X(:,1),X(:,2),X(:,3))");
    mat.eval("hold on");
    mat.eval("scatter3(Y(:,1),Y(:,2),Y(:,3))");
    mat.eval("hold on");
    mat.eval("XL = [X(:,1) Y(:,1)]");
    mat.eval("YL = [X(:,2) Y(:,2)]");
    mat.eval("ZL = [X(:,3) Y(:,3)]");
    mat.eval("plot3(XL',YL',ZL', ':', 'Color', 'black')");
    if (holdOn)
      mat.eval("hold on");
  };
  
  // Read test data set
  const Matrix Y = parseXYZData("bunny.xyz");

  // Rigid ECPD
  std::cout << "Rigid case" << std::endl;
  {
    // Apply some known rigid transformation
    const Matrix R = AngleAxis(Pi_4, Vector3::UnitZ()).toRotationMatrix(); // rotate 45 deg around z-axis
    const Vector3 t = Vector3(1, 2, 3); // shift
    const ScalarType s = 4; // scale
    Matrix X = s * Y * R.transpose() + Vector::Ones(Y.rows()) * t.transpose();
    // Permutate rows. i.e. destroy correspondences, to make it a bit harder
    X = permutateRows(X);

    // Let's see if we can find it
    const auto res = ECPD<Rigid>::compute(X, Y, ECPD<Rigid>::Config()); 
    std::cout << "R = \n" << res.first.R << std::endl; // found rotation matrix
    std::cout << "t = " << res.first.t.transpose() << std::endl; // found translation vector
    std::cout << "s = " << s << std::endl; // found scaling factor
  }
  std::cout << std::endl;

  // Affine ECPD
  std::cout << "Affine case" << std::endl;
  {
    // Apply some known affine transformation
    const Matrix B = Vector3(1, 2, 3).asDiagonal(); // scale/sheer
    const Vector3 t = Vector3(4, 5, 6); // shift 
    Matrix X = Y * B.transpose() + Vector::Ones(Y.rows()) * t.transpose();
    // Permutate rows. i.e. destroy correspondences, to make it a bit harder
    X = permutateRows(X);

    // Let's see if we can find it
    const auto res = ECPD<Affine>::compute(X, Y, ECPD<Affine>::Config());
    std::cout << "B = \n" << res.first.B << std::endl; // found affine matrix
    std::cout << "t = " << res.first.t.transpose() << std::endl; // found translation vector
  }
  std::cout << std::endl;

  // Non-rigid ECPD
  std::cout << "Non-rigid case" << std::endl;
  {
    // Apply some random non-rigid transformation
    Matrix X = Y;
    const Matrix W = 0.005 * Matrix::Random(X.rows(), X.cols());
    X += W + Vector::Ones(Y.rows()) * Vector3(0, 0, 0.1).transpose();
    // Permutate rows. i.e. destroy correspondences, to make it a bit harder
    X = permutateRows(X);

    // Let's see if we can find it
    const auto res = ECPD<NonRigid>::compute(X, Y, ECPD<NonRigid>::Config());
    showPCs(res.second.T, Y);
  }
  std::cout << std::endl;

  char ch;
  std::cin >> ch;
  return 0;
}
