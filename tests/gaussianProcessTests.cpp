#include <iostream>
#include <stdlib.h>


#include "Eigen/Dense"
#include "smlt/gaussianProcess.hpp"
#include "smlt/smltUtilities.hpp"

using namespace smlt;

int main(int argc, char const *argv[])
{
    gaussianProcess gProcess;

    Eigen::MatrixXd inputData(1,3);
    inputData << 0.0, 3.5, 0.01;


    Eigen::MatrixXd outputSamples(3,3);
    outputSamples <<    1.5,     1.2,   1.5,
                        -1.0,    -0.41, -1.0,
                        2.5,     2.9,   2.5;


    Eigen::MatrixXd Sigma = calculateCovariance(inputData, true)*10.0;
    Eigen::VectorXd maxCov = getVariance(Eigen::MatrixXd(outputSamples.transpose()));
    // Eigen::VectorXd maxCov = Eigen::VectorXd::Ones(3);

    std::cout << "\ninputData\n" << inputData << std::endl;
    std::cout << "\noutputSamples\n" << outputSamples << std::endl;
    std::cout << "\ncovariance matrix\n" << Sigma << std::endl;
    std::cout << "\nmaximum covariance\n" << maxCov << std::endl;

    gProcess.setKernelCenters(inputData);
    gProcess.setKernelTrainingData(outputSamples);
    gProcess.setCovarianceMatrix(Sigma);
    gProcess.setMaxCovariance(maxCov);

    std::cout << "Calulating designMatrix and kernelWeights." << std::endl;
    gProcess.calculateParameters();


    std::string dirPath = "/home/ryan/Code/smlt/tmp1/";
    bool overwrite = true;
    std::cout << "Writing kernel data to: " << dirPath << std::endl;
    gProcess.writeDataToFile(dirPath, overwrite);
    gProcess.writeOutputToFile(dirPath, overwrite);


    std::system(("python ./scripts/plotGaussianProcess.py " + dirPath).c_str());


    return 0;

    // Eigen::MatrixXd meanMat, varMat;
    //
    // Eigen::VectorXd testVec = Eigen::VectorXd::Random(2);
    // std::cout << "Testing calculateMeanAndVariance for one input vector." << std::endl;
    //
    // gProcess.calculateMeanAndVariance(testVec, meanMat, varMat);
    // std::cout << "mean = \n" << meanMat << "\nvar = \n" << varMat << std::endl;
    //
    //
    // std::cout << "Testing calculateMeanAndVariance for multiple input vectors." << std::endl;
    // Eigen::MatrixXd testMat = Eigen::MatrixXd::Random(2,3);
    // gProcess.calculateMeanAndVariance(testMat, meanMat, varMat);
    // std::cout << "mean = \n" << meanMat << "\nvar = \n" << varMat << std::endl;
    //
    //
    //
    // std::cout << "Adding kernel data. New centers:" << std::endl;
    // int newDataDim = 10;
    // Eigen::MatrixXd newCenters = Eigen::MatrixXd::Random(2,newDataDim);
    // std::cout << newCenters << std::endl;
    // Eigen::MatrixXd newTrainingData = Eigen::MatrixXd::Random(3,newDataDim);
    // std::cout << "\nNew training data:\n" << newTrainingData << std::endl;
    //
    //
    // gProcess.addNewKernelData(newCenters, newTrainingData);
    //
    // std::cout << "Testing new data for multiple input vectors." << std::endl;
    // gProcess.calculateMeanAndVariance(testMat, meanMat, varMat);
    // std::cout << "mean = \n" << meanMat << "\nvar = \n" << varMat << std::endl;
    //
    //
    // std::cout << "\n\nRemoving newly added kernels..." << std::endl;
    // gProcess.removeRecentlyAddedKernelData();
    // std::cout << "Testing after removal for multiple input vectors." << std::endl;
    // gProcess.calculateMeanAndVariance(testMat, meanMat, varMat);
    // std::cout << "mean = \n" << meanMat << "\nvar = \n" << varMat << std::endl;



}
