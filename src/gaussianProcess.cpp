#include "smlt/gaussianProcess.hpp"

using namespace smlt;

gaussianProcess::gaussianProcess()
{
    kernelFuncPtr = new squaredExponential();
}


void gaussianProcess::train(const Eigen::VectorXd& input, const Eigen::VectorXd& output)
{
    train(Eigen::MatrixXd(input), Eigen::MatrixXd(output));
}

void gaussianProcess::train(const Eigen::VectorXd& input, const Eigen::MatrixXd& output)
{
    train(Eigen::MatrixXd(input), output);
}

void gaussianProcess::train(const Eigen::MatrixXd& input, const Eigen::VectorXd& output)
{
    train(input, Eigen::MatrixXd(output));
}


/*!
*
*   @param [in] input A matrix where each column represents a sample. Size = [Input Dimension x Nsamples]
*   @param [in] output A matrix of outputs for each input column. Size = [Output Dimension x Nsamples]
*/
void gaussianProcess::train(const Eigen::MatrixXd& input, const Eigen::MatrixXd& output)
{
    if (input.cols() == output.cols())
    {
        setKernelCenters(input);
        setKernelTrainingData(output);

        bool onlyDiagonal = true;
        setCovarianceMatrix(calculateCovariance(kernelCenters, onlyDiagonal));
        setMaxCovariance(getVariance(kernelTrainingData));

        calculateParameters();
    }
    else
    {
        smltError("Input training data and output training data do not have equal column dimensions: Input = "<< input.rows() << "x" << input.cols() << " & Output = "<< output.rows() << "x" << output.cols() << "." );
    }

}

void gaussianProcess::setKernelCenters(const Eigen::MatrixXd& newCenters)
{
    kernelCenters = newCenters;
    kernelFuncPtr->setCenters(kernelCenters);
}

void gaussianProcess::setKernelTrainingData(const Eigen::MatrixXd& newTrainingData)
{
    kernelTrainingData = newTrainingData.transpose();
    outputDimension = kernelTrainingData.cols();
    kernelFuncPtr->setCenters(kernelCenters);
}


void gaussianProcess::calculateParameters()
{
    if (kernelCenters.cols() == kernelTrainingData.rows())
    {
        kernelFuncPtr->getDesignMatrix(designMatrix);
        designMatrixInv = designMatrix.inverse();
        kernelWeights = designMatrixInv * kernelTrainingData;
    }
    else
    {
        smltError("The number of kernel centers: " << kernelCenters.cols() << " does not match the number of training data samples: " << kernelTrainingData.rows() << ". Every center should have a coresponding training data sample.");
    }

}


void gaussianProcess::setMaxCovariance(const double maxCovScalar)
{
    setMaxCovariance(Eigen::VectorXd::Constant(outputDimension, maxCovScalar));
}

void gaussianProcess::setMaxCovariance(const Eigen::VectorXd& maxCovVector)
{
    maximumCovariance = maxCovVector;
}


void gaussianProcess::setCovarianceMatrix(const Eigen::VectorXd& userSigmaVec)
{
    kernelFuncPtr->setCovarianceMatrix(userSigmaVec);
}

void gaussianProcess::setCovarianceMatrix(const Eigen::MatrixXd& userSigmaMat)
{
    kernelFuncPtr->setCovarianceMatrix(userSigmaMat);
}


void gaussianProcess::getMeanAndVariance(const Eigen::VectorXd& inputVector, Eigen::MatrixXd& mean, Eigen::MatrixXd& variance)
{
    bool check1 = kernelCenters.cols() == kernelTrainingData.rows();
    bool check2 = outputDimension == kernelWeights.cols();

    if (check1 && check2)
    {
        Eigen::MatrixXd kernelOutput;
        kernelFuncPtr->evaluate(inputVector, kernelOutput);

        mean.resize(outputDimension,1);
        variance.resize(outputDimension,1);

        mean = (kernelOutput * kernelWeights).transpose(); // Kern out = [1 x Nc], kernweights = [Nc, outputDimension]

        variance = maximumCovariance - (maximumCovariance.array().square() * (kernelOutput * designMatrixInv * kernelOutput.transpose()).replicate(outputDimension,1).array()).matrix();    }
    else
    {
        if (!check1) {
            smltError("The number of kernel centers: " << kernelCenters.cols() << " does not match the number of training data samples: " << kernelTrainingData.rows() << ". Every center should have a coresponding training data sample.");
        }
        if (!check2) {
            smltError("The expected output dimension: " << outputDimension << " does not match the number of kernel weights: " << kernelWeights.cols() << ". Make sure to retrain when new data is added.");
        }
    }

}

void gaussianProcess::getMeanAndVariance(const Eigen::MatrixXd& inputVectors, Eigen::MatrixXd& means, Eigen::MatrixXd& variances)
{
    bool check1 = kernelCenters.cols() == kernelTrainingData.rows();
    bool check2 = outputDimension == kernelWeights.cols();

    if (check1 && check2)
    {
        int numberOfInputs = inputVectors.cols();
        Eigen::MatrixXd kernelOutput;

        kernelFuncPtr->evaluate(inputVectors, kernelOutput);

        means.resize(outputDimension, numberOfInputs);
        variances.resize(outputDimension, numberOfInputs);

        means = (kernelOutput * kernelWeights).transpose(); // Kern out = [Ns x Nc], kernweights = [Nc, outputDimension]

        kernelOutput.transposeInPlace();

        variances = maximumCovariance.replicate(1,numberOfInputs) -  (maximumCovariance.array().square().replicate(1,numberOfInputs).array() * ((kernelOutput.array() * (designMatrixInv * kernelOutput).array()).colwise().sum() ).replicate(outputDimension,1).array()).matrix();
    }
    else
    {
        if (!check1) {
            smltError("The number of kernel centers: " << kernelCenters.cols() << " does not match the number of training data samples: " << kernelTrainingData.rows() << ". Every center should have a coresponding training data sample.");
        }
        if (!check2) {
            smltError("The expected output dimension: " << outputDimension << " does not match the number of kernel weights: " << kernelWeights.cols() << ". Make sure to retrain when new data is added.");
        }
    }

}




void gaussianProcess::addNewKernelData(const Eigen::VectorXd& newCenters, const Eigen::VectorXd& newTrainingData)
{
    addNewKernelData(Eigen::MatrixXd(newCenters), Eigen::MatrixXd(newTrainingData));
}

void gaussianProcess::addNewKernelData(const Eigen::VectorXd& newCenters, const Eigen::MatrixXd& newTrainingData)
{
    addNewKernelData(Eigen::MatrixXd(newCenters), newTrainingData);
}

void gaussianProcess::addNewKernelData(const Eigen::MatrixXd& newCenters, const Eigen::VectorXd& newTrainingData)
{
    addNewKernelData(newCenters, Eigen::MatrixXd(newTrainingData));
}

void gaussianProcess::addNewKernelData(const Eigen::MatrixXd& newCenters, const Eigen::MatrixXd& newTrainingData)
{
    bool check1 = newCenters.cols() == newTrainingData.cols();
    bool check2 = newCenters.rows() == kernelCenters.rows();
    bool check3 = newTrainingData.rows() == kernelTrainingData.cols();

    if (check1 && check2 && check3)
    {
        newKernelDataStartIndex = kernelCenters.cols();
        newKernelDataSize = newCenters.cols();
        Eigen::MatrixXd newCentersMat = Eigen::MatrixXd::Zero(kernelCenters.rows(), newKernelDataStartIndex+newKernelDataSize);

        newCentersMat.leftCols(newKernelDataStartIndex) = kernelCenters;
        newCentersMat.rightCols(newKernelDataSize) = newCenters;
        setKernelCenters(newCentersMat);

        Eigen::MatrixXd newTrainingDataMat = Eigen::MatrixXd::Zero(kernelTrainingData.cols(), kernelTrainingData.rows()+newKernelDataSize);

        newTrainingDataMat.leftCols(kernelTrainingData.rows()) = kernelTrainingData.transpose();
        newTrainingDataMat.rightCols(newKernelDataSize) = newTrainingData;
        setKernelTrainingData(newTrainingDataMat);

        calculateParameters();

    }
    else
    {
        if (!check1) {
            smltError("The number of kernel centers: " << newCenters.cols() << " does not match the number of training data samples: " << newTrainingData.rows() << ". Every center should have a coresponding training data sample.");
        }
        if (!check2) {
            smltError("The kernel center dimension: " << newCenters.rows() << " does not match the current kernel center dimension: " << kernelCenters.rows() << ". All new kernel centers must have the same dimension as existing kernel centers.");
        }
        if (!check3) {
            smltError("The training data dimension: " << newTrainingData.rows() << " does not match the current training data dimension: " << kernelTrainingData.cols() << ". All new training data must have the same dimension as existing training data.");
        }
    }
}


void gaussianProcess::removeKernelData(const int startIndex, const int nKernToRemove)
{
    bool check1 = (startIndex>=0) && (startIndex<kernelCenters.cols());
    bool check2 = (startIndex+nKernToRemove)<=kernelCenters.cols();
    if (check1 && check2) {

        removeCols(kernelCenters, startIndex, nKernToRemove);
        removeRows(kernelTrainingData, startIndex, nKernToRemove);

        setKernelCenters(kernelCenters);
        setKernelTrainingData(kernelTrainingData.transpose());
        calculateParameters();
    }
    else {
        if (!check1) {
            smltError("Start index (" << startIndex << ") is outside of bounds [0:"<< kernelCenters.cols()-1 <<"]. Please select a starting index for removal that is within these bounds.");
        }
        if (!check2) {
            smltError("You are trying to remove kernel indexes ["<< startIndex<<":"<< startIndex+nKernToRemove-1 <<"] which exceed the current kernel dimension: [0:"<< kernelCenters.cols()-1 <<"].");
        }
    }

}

void gaussianProcess::removeRecentlyAddedKernelData()
{
    removeKernelData(newKernelDataStartIndex, newKernelDataSize);
}

void gaussianProcess::writeDataToFile(std::string directoryPath, const bool overwrite)
{
    directoryPath = "/" + directoryPath + "/";
    checkAndCreateDirectory(directoryPath);
    directoryPath += "gaussianProcessKernelData/";
    checkAndCreateDirectory(directoryPath);

    std::ofstream kernelCentersFile, kernelTrainingDataFile, maximumCovarianceFile, covarianceMatrixFile;

    std::string kernelCentersPath       = directoryPath+"/"+"kernelCenters.txt";
    std::string kernelTrainingDataPath  = directoryPath+"/"+"kernelTrainingData.txt";
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
    kernelTrainingDataFile.open(kernelTrainingDataPath.c_str(), mode);
    maximumCovarianceFile.open(maximumCovariancePath.c_str(), mode);
    covarianceMatrixFile.open(covarianceMatrixPath.c_str(), mode);


    kernelCentersFile << kernelCenters;
    kernelTrainingDataFile << kernelTrainingData;
    maximumCovarianceFile << maximumCovariance;
    covarianceMatrixFile << kernelFuncPtr->getCovarianceMatrix();


    kernelCentersFile.close();
    kernelTrainingDataFile.close();
    maximumCovarianceFile.close();
    covarianceMatrixFile.close();

    std::cout << "\nKernel data saved to: " << directoryPath << std::endl;
}


void gaussianProcess::writeOutputToFile(std::string directoryPath, const bool overwrite)
{
    directoryPath = "/" + directoryPath + "/";
    checkAndCreateDirectory(directoryPath);
    directoryPath += "gaussianProcessOutput/";
    checkAndCreateDirectory(directoryPath);

    std::ofstream inputFile, kernelOutputFile, gaussianProcessMeanFile, gaussianProcessVarianceFile, inputMinMaxFile;

    std::string inputPath                       = directoryPath+"/"+"input.txt";
    std::string kernelOutputPath                = directoryPath+"/"+"kernelOutput.txt";
    std::string gaussianProcessMeanPath         = directoryPath+"/"+"gaussianProcessMean.txt";
    std::string gaussianProcessVariancePath     = directoryPath+"/"+"gaussianProcessVariance.txt";
    std::string inputMinMaxPath                 = directoryPath+"/"+"inputMinMax.txt";

    std::ios_base::openmode mode;
    if (overwrite) {
        mode = std::ios_base::trunc;
    }
    else {
        mode = std::ios_base::app;
    }

    inputFile.open(inputPath.c_str(), mode);
    kernelOutputFile.open(kernelOutputPath.c_str(), mode);
    gaussianProcessMeanFile.open(gaussianProcessMeanPath.c_str(), mode);
    gaussianProcessVarianceFile.open(gaussianProcessVariancePath.c_str(), mode);
    inputMinMaxFile.open(inputMinMaxPath.c_str(), mode);

    Eigen::VectorXd mins = kernelCenters.rowwise().minCoeff();
    Eigen::VectorXd maxs = kernelCenters.rowwise().maxCoeff();
    int nSteps = 100;

    Eigen::MatrixXd input = discretizeSearchSpace(mins, maxs, nSteps);

    Eigen::MatrixXd kernelOutput, gpMean, gpVariance;
    kernelFuncPtr->evaluate(input, kernelOutput);

    getMeanAndVariance(input, gpMean, gpVariance);

    inputFile << input;
    kernelOutputFile << kernelOutput;
    gaussianProcessMeanFile << gpMean;
    gaussianProcessVarianceFile << gpVariance;
    inputMinMaxFile << mins.transpose() << std::endl << maxs.transpose();


    inputFile.close();
    kernelOutputFile.close();
    gaussianProcessMeanFile.close();
    gaussianProcessVarianceFile.close();
    inputMinMaxFile.close();

    std::cout << "\nGaussian process output saved to: " << directoryPath << std::endl;
}
