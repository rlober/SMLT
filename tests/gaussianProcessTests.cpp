#include <iostream>
#include "Eigen/Dense"
#include "smlt/gaussianProcess.hpp"
#include "smlt/smltUtilities.hpp"

using namespace smlt;

int main(int argc, char const *argv[])
{
    gaussianProcess gProcess;

    Eigen::MatrixXd inputData(2,2);
    inputData << -1, 2,
                 -3, 4;

    Eigen::MatrixXd outputSamples;

    Eigen::MatrixXd outputSamples1(1,2);
    outputSamples1 << 1.5, 2.5;


    Eigen::MatrixXd outputSamples2(3,2);
    outputSamples2 <<   1.5, 0.1,
                        -1.0, -0.2,
                        2.5, 3.1;


    outputSamples = outputSamples2;


    std::cout << "inputData\n" << inputData << std::endl;
    std::cout << "outputSamples\n" << outputSamples << std::endl;

    std::cout << "Setting kernel centers." << std::endl;
    gProcess.setKernelCenters(inputData);

    std::cout << "Setting training data to:\n" << outputSamples << std::endl;
    gProcess.setKernelTrainingData(outputSamples);

    std::cout << "Setting covariance matrix to:\n" << calculateCovariance(inputData, true) << std::endl;
    gProcess.setCovarianceMatrix(calculateCovariance(inputData, true));

    std::cout << "Setting maximum covariance to:\n" << getVariance(Eigen::MatrixXd(outputSamples.transpose())) << std::endl;
    gProcess.setMaxCovariance(getVariance(Eigen::MatrixXd(outputSamples.transpose())));

    std::cout << "Calulating designMatrix and kernelWeights." << std::endl;
    gProcess.calculateParameters();

    Eigen::MatrixXd meanMat, varMat;

    Eigen::VectorXd testVec = Eigen::VectorXd::Random(2);
    std::cout << "Testing getMeanAndVariance for one input vector." << std::endl;

    gProcess.getMeanAndVariance(testVec, meanMat, varMat);
    std::cout << "mean = \n" << meanMat << "\nvar = \n" << varMat << std::endl;


    std::cout << "Testing getMeanAndVariance for multiple input vectors." << std::endl;
    Eigen::MatrixXd testMat = Eigen::MatrixXd::Random(2,3);
    gProcess.getMeanAndVariance(testMat, meanMat, varMat);
    std::cout << "mean = \n" << meanMat << "\nvar = \n" << varMat << std::endl;



    std::cout << "Adding kernel data. New centers:" << std::endl;
    int newDataDim = 10;
    Eigen::MatrixXd newCenters = Eigen::MatrixXd::Random(2,newDataDim);
    std::cout << newCenters << std::endl;
    Eigen::MatrixXd newTrainingData = Eigen::MatrixXd::Random(3,newDataDim);
    std::cout << "\nNew training data:\n" << newTrainingData << std::endl;


    gProcess.addNewKernelData(newCenters, newTrainingData);

    std::cout << "Testing new data for multiple input vectors." << std::endl;
    gProcess.getMeanAndVariance(testMat, meanMat, varMat);
    std::cout << "mean = \n" << meanMat << "\nvar = \n" << varMat << std::endl;


    std::cout << "\n\nRemoving newly added kernels..." << std::endl;
    gProcess.removeRecentlyAddedKernelData();
    std::cout << "Testing after removal for multiple input vectors." << std::endl;
    gProcess.getMeanAndVariance(testMat, meanMat, varMat);
    std::cout << "mean = \n" << meanMat << "\nvar = \n" << varMat << std::endl;


    std::string dirPath = "/home/ryan/Code/smlt/tmp1/";
    bool overwrite = true;
    std::cout << "Writing kernel data to: " << dirPath << std::endl;
    gProcess.writeDataToFile(dirPath, overwrite);
    gProcess.writeOutputToFile(dirPath, overwrite);

    return 0;
}
