#include <iostream>
#include "Eigen/Dense"
#include "smlt/smltUtilities.hpp"

using namespace smlt;

int main(int argc, char const *argv[])
{
    // smltError( "This is a test error message...");
    //
    // smltWarning("This is a test warning message...");
    //
    // Eigen::MatrixXd testMat(2,3);
    // testMat << 1, 2, 0,
    //            3, 4, 2;
    //
    // std::cout << "testMat is:\n" << testMat << std::endl;
    // std::cout << "Its covariance matrix is:\n" << calculateCovariance(testMat) << std::endl;
    //
    // Eigen::MatrixXd m = Eigen::MatrixXd::Random(3,5);
    // std::cout << "m: \n" << m << std::endl;
    //
    // std::cout << "\nRemove 2 columns starting at index 2:" << std::endl;
    // removeCols(m, 2, 2);
    // std::cout << m << std::endl;
    //
    // std::cout << "\nRemove 2 rows starting at index 1:" << std::endl;
    // removeRows(m, 1, 2);
    // std::cout << m << std::endl;
    //
    // Eigen::VectorXd v = Eigen::VectorXd::Random(5);
    // std::cout << "v: \n" << v << std::endl;
    // std::cout << "\nRemove 2 rows starting at index 1:" << std::endl;
    // removeRows(v, 1, 2);
    // std::cout << v << std::endl;
    //
    //
    //
    // std::cout << "\n\n\n" << std::endl;
    //
    //
    // std::string dirPath = "../../tmp_dir";
    // std::cout << "Create directory: " << dirPath << std::endl;

    // checkAndCreateDirectory(dirPath);


    std::cout << "\n\nChecking discretizeSearchSpace()" << std::endl;
    Eigen::VectorXd minVals = Eigen::VectorXd::Zero(3);
    Eigen::VectorXd maxVals = Eigen::VectorXd::Ones(3);
    Eigen::VectorXi nStepVec(3);
    nStepVec << 3, 5, 2;
    std::cout << "minVals:\n" << minVals << std::endl;
    std::cout << "maxVals:\n" << maxVals << std::endl;
    std::cout << "nStepVec: " << nStepVec << std::endl;
    Eigen::MatrixXd result;
    discretizeSearchSpace(minVals, maxVals, nStepVec, result);
    std::cout << "result: "<< result.rows() << "x" << result.cols() << "\n" << result.transpose() << std::endl;


    int nStep = 4;
    std::cout << "nStep: " << nStep << std::endl;
    discretizeSearchSpace(minVals, maxVals, nStep, result);
    std::cout << "result: "<< result.rows() << "x" << result.cols() << "\n" << result.transpose() << std::endl;

    // std::cout << "\nresult:\n" << result<< std::endl;


    return 0;
}
