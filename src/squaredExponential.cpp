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
        // Eigen::MatrixXd inputMat = inputVector.replicate(1, numberOfKernels); // Repeat the input column vector numberOfKernels times along the column axis.
        // inputMat -= kernelCenters; // perform (x-c).

        Eigen::MatrixXd dummy1 = (kernelCenters.colwise() - inputVector).array().square();
        Eigen::MatrixXd dummy2 = dummy1.array() /  (-0.5 * sigmaMat.diagonal()).replicate(1,numberOfKernels).array();
        Eigen::MatrixXd inputMat = dummy2.colwise().sum();


        output.resize(1, numberOfKernels);
        // output = maximumCovariance * (-0.5 * (inputMat.array() * (sigmaMatInv * inputMat).array()).colwise().sum() ).array().exp();
        output = maximumCovariance * (inputMat).array().exp();

    }
    else{
        smltError("Input vector does not match kernel dimension. Returning 0's.");
        output = Eigen::MatrixXd::Zero(1, numberOfKernels);
    }
}

void squaredExponential::doEvaluate(const Eigen::MatrixXd& _inputVectors, Eigen::MatrixXd& output)
{

    if (_inputVectors.cols()==1)
    {
        doEvaluate(Eigen::VectorXd(_inputVectors), output);
    }

    else
    {
        // Copy over input vectors so we can modify
        Eigen::MatrixXd inputVectors = _inputVectors;


        int numberOfInputs = inputVectors.cols();

        // Check that the input vectors are the right dimension
        if (inputVectors.rows() == kernelDimension)
        {
// time_t c0 = std::time(0);
            // Get the number of inputs
            int inputSize = kernelDimension*numberOfInputs;

// time_t c1 = std::time(0);

            // First flatten the input vectors into a big column
            inputVectors.resize(inputSize,1);

// time_t c2 = std::time(0);


            // Then repeat the input column vector numberOfKernels times along the column axis.
            // Eigen::MatrixXd inputMat = inputVectors.replicate(1, numberOfKernels);


// time_t c3 = std::time(0);


            // Precalculate (X-C)
            // inputMat -= kernelCenters.replicate(numberOfInputs,1);


// time_t c4 = std::time(0);


            // Create a summation matrix which is essentially just a block diagonal matrix of one vectors which avoids having to explicitly use a for loop to sum kernel outputs for a single input.
            // std::clock_t cl = std::clock();

            // Eigen::MatrixXd summationMatrix = blockDiag(Eigen::MatrixXd::Ones(1, kernelDimension), numberOfInputs);

// time_t c5 = std::time(0);


            // Repeat the covariance matrix's inverse numberOfInputs times.
            // Eigen::MatrixXd sigmaMatInvBlk = blockDiag(sigmaMatInv, numberOfInputs);
            // std::clock_t cl0 = std::clock();
            // std::cout << "Test 1" << std::endl;
            Eigen::MatrixXd dummy1 = (kernelCenters.replicate(numberOfInputs,1).colwise() - Eigen::VectorXd(inputVectors)).array().square();

            // std::clock_t cl1 = std::clock();

            // std::cout << "Test 2" << std::endl;
            Eigen::MatrixXd dummy2 = dummy1.array() /  (-0.5 * sigmaMat.diagonal()).replicate(numberOfInputs,numberOfKernels).array();

            // std::clock_t cl2 = std::clock();

            // std::cout << "Test 3" << std::endl;
            // Eigen::MatrixXd inputMat = summationMatrix * dummy2;
            // std::cout << "dummy2 before resize\n" << dummy2 << "\n\n";

            dummy2.resize(kernelDimension, numberOfInputs*numberOfKernels);
            // std::cout << "dummy2 after resize 1\n" << dummy2 << "\n\n";
            Eigen::MatrixXd inputMat = dummy2.colwise().sum();
            // std::cout << "inputMat colwise sum\n" << inputMat << "\n\n";
            inputMat.resize(numberOfInputs, numberOfKernels);
            // std::cout << "inputMat after resize 2\n" << inputMat << "\n\n";

            // std::clock_t cl3 = std::clock();

            // std::cout << "Test 4" << std::endl;

            // Eigen::MatrixXd inputMat = summationMatrix * ((kernelCenters.replicate(numberOfInputs,1).colwise() - Eigen::VectorXd(inputVectors)).array().square().array() /  (-0.5 * sigmaMat.diagonal()).replicate(numberOfInputs,numberOfKernels).array() ).matrix();

// time_t c6 = std::time(0);

            // Prepare the output matrix.
            output.resize(numberOfInputs, numberOfKernels);


// time_t c7 = std::time(0);

            // Calculate the vectorized squared exponential kernel function.
            // output = maximumCovariance * (-0.5 * (summationMatrix* (inputMat.array() * (sigmaMatInvBlk * inputMat).array()).matrix()) ).array().exp();
            output = maximumCovariance * (inputMat).array().exp();







// time_t c8 = std::time(0);

            // std::cout << "---------------------------------------------------------------" << std::endl;

            // std::cout << "c0 = " << c0 << std::endl;
            // std::cout << "c1 = " << c1 << std::endl;
            // std::cout << "c2 = " << c2 << std::endl;
            // std::cout << "c3 = " << c3 << std::endl;
            // std::cout << "c4 = " << c4 << std::endl;
            // std::cout << "c5 = " << c5 << std::endl;
            // std::cout << "c6 = " << c6 << std::endl;
            // std::cout << "c7 = " << c7 << std::endl;
            // std::cout << "c8 = " << c8 << std::endl;
            //
            // std::cout << "line: int inputSize = kernelDimension*numberOfInputs \ntime elapsed: " << difftime(c1, c0) * 1000.0 << "\n\n";
            // std::cout << "line: inputVectors.resize(inputSize,1) \ntime elapsed: " << difftime(c2, c1) * 1000.0 << "\n\n";
            // std::cout << "line: Eigen::MatrixXd inputMat = inputVectors.replicate(1, numberOfKernels) \ntime elapsed: " << difftime(c3, c2) * 1000.0 << "\n\n";
            // std::cout << "line: inputMat -= kernelCenters.replicate(numberOfInputs,1) \ntime elapsed: " << difftime(c4, c3) * 1000.0 << "\n\n";
            // std::cout << "line: Eigen::MatrixXd summationMatrix = blockDiag(Eigen::MatrixXd::Ones(1, kernelDimension), numberOfInputs) \ntime elapsed: " << difftime(c5, c4) * 1000.0 << "\n\n";
            // std::cout << "line: Eigen::MatrixXd sigmaMatInvBlk = blockDiag(sigmaMatInv, numberOfInputs) \ntime elapsed: " << difftime(c6, c5) * 1000.0 << "\n\n";
            // std::cout << "line: output.resize(numberOfInputs, numberOfKernels) \ntime elapsed: " << difftime(c7, c6) * 1000.0 << "\n\n";
            // std::cout << "line: output = maximumCovariance * (-0.5 * (summationMatrix* (inputMat.array() * (sigmaMatInvBlk * inputMat).array()).matrix()) ).array().exp() \ntime elapsed: " << difftime(c8, c7) * 1000.0 << "\n\n";

            // std::cout << "line: Test 0 \ntime elapsed: " << (cl0 - cl) / static_cast<double>( CLOCKS_PER_SEC ) << "\n\n";
            // std::cout << "line: Test 1 \ntime elapsed: " << (cl1 - cl0) / static_cast<double>( CLOCKS_PER_SEC ) << "\n\n";
            // std::cout << "line: Test 2 \ntime elapsed: " << (cl2 - cl1) / static_cast<double>( CLOCKS_PER_SEC ) << "\n\n";
            // std::cout << "line: Test 3 \ntime elapsed: " << (cl3 - cl2) / static_cast<double>( CLOCKS_PER_SEC ) << "\n\n";



            // std::cout << "line: summationMatrix \ntime elapsed: " << difftime(c5, c4) << "\n\n";
            // std::cout << "line: inputMat \ntime elapsed: " << difftime(c6, c5) << "\n\n";
            // std::cout << "line: output \ntime elapsed: " << difftime(c8, c7) << "\n\n";
            // std::cout << "---------------------------------------------------------------" << std::endl;



        }
        else{
            smltError("Input vector does not match kernel dimension. Returning 0's.");
            output = Eigen::MatrixXd::Zero(numberOfInputs, numberOfKernels);
        }
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

    // mins -= (mins*0.3);
    // maxs += (maxs*0.3);

    for(int i=0; i<mins.rows(); i++)
    {
        if (mins(i)<0) {
            mins(i) *= 1.3;
        }else{
            mins(i) *= 0.7;
        }
        if (maxs(i)>0) {
            maxs(i) *= 1.3;
        }else{
            maxs(i) *= 0.7;
        }
    }
    int nSteps;

    if (kernelDimension<=3) {
        nSteps = 50;
    }else{
        nSteps = 8;
    }

    Eigen::MatrixXd kernelInput;
    discretizeSearchSpace(mins, maxs, nSteps, kernelInput);

    Eigen::MatrixXd kernelOutput;

    std::cout << "\nEvaluating kernels over the discretized search space. This could take a minute..." << std::endl;
    evaluate(kernelInput, kernelOutput);
    std::cout << "Done!" << std::endl;



    kernelInputFile << kernelInput;
    kernelOutputFile << kernelOutput;
    inputMinMaxFile << mins.transpose() << std::endl << maxs.transpose();


    kernelInputFile.close();
    kernelOutputFile.close();
    inputMinMaxFile.close();

    std::cout << "\nKernel data saved to: " << directoryPath << std::endl;
}
