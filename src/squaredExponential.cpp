#include "smlt/squaredExponential.hpp"

using namespace smlt;

squaredExponential::squaredExponential()
{
    setMaxCovariance(1.0);
}


squaredExponential::squaredExponential(const Eigen::MatrixXd& centers) :
kernelFunction(centers)
{
    // Do stuff?
    setMaxCovariance(1.0);
    setCovarianceMatrix(calculateCovariance(kernelCenters));
}

void squaredExponential::doInit()
{
    std::cout << "beep" << std::endl;
}

void squaredExponential::setMaxCovariance(const double maxCovScalar)
{
    maximumCovariance = maxCovScalar;
}

bool squaredExponential::setCovarianceMatrix(const Eigen::VectorXd& userSigmaVec)
{
    if (userSigmaVec.rows() == kernelDimension) {
        Eigen::MatrixXd diagSigma = userSigmaVec.asDiagonal();
        setCovarianceMatrix(diagSigma);
        return true;
    }
    else{
        smltError("The variance vector provided does not match the dimension of the kernels.");
        return false;
    }

}

bool squaredExponential::setCovarianceMatrix(const Eigen::MatrixXd& userSigmaMat)
{
    if (userSigmaMat.rows() == kernelDimension) {
        sigmaMat = userSigmaMat;
        sigmaMatInv = sigmaMat.inverse();
        return true;
    }
    else{
        smltError("The covariance matrix provided does not match the dimension of the kernels.");
        return false;
    }
}

Eigen::MatrixXd squaredExponential::getCovarianceMatrix()
{
    return sigmaMat;
}

void squaredExponential::doEvaluate(const Eigen::VectorXd& inputVector, Eigen::MatrixXd& output)
{
    if (inputVector.rows() == kernelDimension)
    {
        Eigen::MatrixXd inputMat = inputVector.replicate(1, numberOfKernels); // Repeat the input column vector numberOfKernels times along the column axis.
        inputMat -= kernelCenters; // perform (x-c).

        output.resize(1, numberOfKernels);
        output = maximumCovariance * (-0.5 * (inputMat.array() * (sigmaMatInv * inputMat).array()).colwise().sum() ).array().exp();
    }
    else{
        smltError("Input vector does not match kernel dimension. Returning 0's.");
        output = Eigen::MatrixXd::Zero(1, numberOfKernels);
    }
}

void squaredExponential::doEvaluate(const Eigen::MatrixXd& _inputVectors, Eigen::MatrixXd& output)
{
    // Copy over input vectors so we can modify
    Eigen::MatrixXd inputVectors = _inputVectors;

    int numberOfInputs = inputVectors.cols();

    // Check that the input vectors are the right dimension
    if (inputVectors.rows() == kernelDimension)
    {
        // Get the number of inputs
        int inputSize = kernelDimension*numberOfInputs;

        // First flatten the input vectors into a big column
        inputVectors.resize(inputSize,1);

        // Then repeat the input column vector numberOfKernels times along the column axis.
        Eigen::MatrixXd inputMat = inputVectors.replicate(1, numberOfKernels);

        // Precalculate (X-C)
        inputMat -= kernelCenters.replicate(numberOfInputs,1);

        // Create a summation matrix which is essentially just a block diagonal matrix of one vectors which avoids having to explicitly use a for loop to sum kernel outputs for a single input.
        Eigen::MatrixXd summationMatrix = blockDiag(Eigen::MatrixXd::Ones(1, kernelDimension), numberOfInputs);

        // Repeat the covariance matrix's inverse numberOfInputs times.
        Eigen::MatrixXd sigmaMatInvBlk = blockDiag(sigmaMatInv, numberOfInputs);

        // Prepare the output matrix.
        output.resize(numberOfInputs, numberOfKernels);

        // std::cout << "inputMat: " << inputMat.rows() << "x" << inputMat.cols() << std::endl;
        // std::cout << "sigmaMatInvBlk: " << sigmaMatInvBlk.rows() << "x" << sigmaMatInvBlk.cols() << std::endl;
        // std::cout << "summationMatrix: " << summationMatrix.rows() << "x" << summationMatrix.cols() << std::endl;
        // std::cout << ": " << .rows() << "x" << .cols() << std::endl;


        // Calculate the vectorized squared exponential kernel function.
        output = maximumCovariance * (-0.5 * (summationMatrix* (inputMat.array() * (sigmaMatInvBlk * inputMat).array()).matrix()) ).array().exp();
    }
    else{
        smltError("Input vector does not match kernel dimension. Returning 0's.");
        output = Eigen::MatrixXd::Zero(numberOfInputs, numberOfKernels);
    }
}

void squaredExponential::writeDataToFile(std::string directoryPath, const bool overwrite)
{
    directoryPath = "/" + directoryPath + "/";
    checkAndCreateDirectory(directoryPath);
    directoryPath += "squaredExponentialKernelData/";
    checkAndCreateDirectory(directoryPath);

    std::ofstream kernelCentersFile, covarianceMatrixFile;

    std::string kernelCentersPath       = directoryPath+"/"+"kernelCenters.txt";
    std::string maximumCovariancePath   = directoryPath+"/"+"maximumCovariance.txt";
    std::string covarianceMatrixPath    = directoryPath+"/"+"covarianceMatrix.txt";

    std::ios_base::openmode mode;
    if (overwrite) {
        mode = std::ios_base::trunc;
    }
    else {
        mode = std::ios_base::app;
    }

    kernelCentersFile.open(kernelCentersPath.c_str(), mode);
    covarianceMatrixFile.open(covarianceMatrixPath.c_str(), mode);


    kernelCentersFile << kernelCenters;
    covarianceMatrixFile << getCovarianceMatrix();


    kernelCentersFile.close();
    covarianceMatrixFile.close();

    std::cout << "\nKernel data saved to: " << directoryPath << std::endl;
}

void squaredExponential::writeOutputToFile(std::string directoryPath, const bool overwrite)
{
    directoryPath = "/" + directoryPath + "/";
    checkAndCreateDirectory(directoryPath);
    directoryPath += "squaredExponentialKernelOutput/";
    checkAndCreateDirectory(directoryPath);

    std::ofstream kernelInputFile, kernelOutputFile, inputMinMaxFile;

    std::string kernelInputPath   = directoryPath+"/"+"kernelInput.txt";
    std::string kernelOutputPath  = directoryPath+"/"+"kernelOutput.txt";
    std::string inputMinMaxPath   = directoryPath+"/"+"inputMinMax.txt";

    std::ios_base::openmode mode;
    if (overwrite) {
        mode = std::ios_base::trunc;
    }
    else {
        mode = std::ios_base::app;
    }

    kernelInputFile.open(kernelInputPath.c_str(), mode);
    kernelOutputFile.open(kernelOutputPath.c_str(), mode);
    inputMinMaxFile.open(inputMinMaxPath.c_str(), mode);

    Eigen::VectorXd mins = kernelCenters.rowwise().minCoeff();
    Eigen::VectorXd maxs = kernelCenters.rowwise().maxCoeff();
    int nSteps = 100;

    Eigen::MatrixXd kernelInput = discretizeSearchSpace(mins, maxs, nSteps);

    Eigen::MatrixXd kernelOutput;
    evaluate(kernelInput, kernelOutput);



    kernelInputFile << kernelInput;
    kernelOutputFile << kernelOutput;
    inputMinMaxFile << mins.transpose() << std::endl << maxs.transpose();


    kernelInputFile.close();
    kernelOutputFile.close();
    inputMinMaxFile.close();

    std::cout << "\nKernel data saved to: " << directoryPath << std::endl;
}
