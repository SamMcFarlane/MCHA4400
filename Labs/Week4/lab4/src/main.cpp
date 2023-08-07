#include <cassert>
#include <cstdlib>
#include <iostream>
#include <Eigen/Core>

int main(int argc, char *argv[])
{
    std::cout << "Eigen version: ";
    std::cout<<EIGEN_WORLD_VERSION<<"."<<EIGEN_MAJOR_VERSION<<"."<<EIGEN_MINOR_VERSION;
    std::cout << "\n" << std::endl;

    std::cout << "Create a column vector:" << std::endl;
    Eigen::VectorXd x(3);
    x<< 1, 3.2, 0.01;
    std::cout << "x = \n" << x << "\n" << std::endl;

    std::cout << "Create a matrix:" << std::endl;
    Eigen::VectorXd i = Eigen::VectorXd::LinSpaced(4, 1, 4);
    // Create a vector of size 3 containing 1, 2, 3
    Eigen::VectorXd j = Eigen::VectorXd::LinSpaced(3, 1, 3);

    // Use broadcasting to create the matrix A without a loop
    Eigen::MatrixXd A = i * j.transpose();

    // TODO: Don't just use a for loop or hardcode all the elements
    //       Try and be creative :)
    std::cout << "A.size() = " << A.size() << std::endl;
    std::cout << "A.rows() = " << A.rows() << std::endl;
    std::cout << "A.cols() = " << A.cols() << std::endl;
    std::cout << "A = \n" << A << "\n" << std::endl;
    std::cout << "A.transpose() = \n" << A.transpose() << "\n" << std::endl;

    std::cout << "Matrix multiplication:" << std::endl;
    Eigen::VectorXd Ax;
    Ax = A*x;
    std::cout << "A*x = \n" << Ax << "\n" << std::endl;

    std::cout << "Matrix concatenation:" << std::endl;
    Eigen::MatrixXd B(4,6);
    B << A, A;
    std::cout << "B = \n" << B << "\n" << std::endl;
    Eigen::MatrixXd C(8,3);
    C << A,A;
    std::cout << "C = \n" << C << "\n" << std::endl;

    std::cout << "Submatrix via block:" << std::endl;
    Eigen::MatrixXd D(1,3);
    D = B.block(1,2,1,3);
    std::cout << "D = \n" << D << "\n" << std::endl;
    std::cout << "Submatrix via slicing:" << std::endl;
    D = B(1,Eigen::seq(2,4));
    std::cout << "D = \n" << D << "\n" << std::endl;

    std::cout << "Broadcasting:" << std::endl;
    Eigen::VectorXd v(6);
    v<< 1,3,5,7,4,6;
    Eigen::MatrixXd E;
    E = B.rowwise() + v.transpose();
    std::cout << "E = \n" << E << "\n" << std::endl;

    std::cout << "Index subscripting:" << std::endl;
    // Define arrays r and c as indices for rows and columns
    Eigen::VectorXi r(4);
    Eigen::VectorXi c(6);
    
    r<<1,3,2,4;
    c<<1, 4, 2, 5, 3, 6;

    r = r.array()-1;
    c = c.array() - 1;
 
    Eigen::MatrixXd F = B(r, c);
 
    std::cout << "F = \n" << F << "\n" << std::endl;

    std::cout << "Memory mapping:" << std::endl;
    float array[9] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
    Eigen::Map<Eigen::Matrix<float,3,3,Eigen::RowMajor>> G(array);              // TODO: Replace this with an Eigen::Map
    array[2] = -3.0f;               // Change an element in the raw storage
    assert(array[2] == G(0,2));     // Ensure the change is reflected in the view
    G(2,0) = -7.0f;                 // Change an element via the view
    assert(G(2,0) == array[6]);     // Ensure the change is reflected in the raw storage
    std::cout << "G = \n" << G << "\n" << std::endl;

    return EXIT_SUCCESS;
}
