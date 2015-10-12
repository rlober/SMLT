#ifndef GAUSSIANPROCESS_HPP
#define GAUSSIANPROCESS_HPP

#include <iostream>
#include <fstream>
#include <stdlib.h>

#include "Eigen/Dense"

#include "smlt/kernelFunction.hpp"
#include "smlt/squaredExponential.hpp"

namespace smlt
{
    class gaussianProcess
    {
    public:
        gaussianProcess();

        void train(const Eigen::VectorXd& input, const Eigen::VectorXd& output);
        void train(const Eigen::VectorXd& input, const Eigen::MatrixXd& output);
        void train(const Eigen::MatrixXd& input, const Eigen::VectorXd& output);
        void train(const Eigen::MatrixXd& input, const Eigen::MatrixXd& output);

        void calculateParameters();

        void setKernelCenters(const Eigen::MatrixXd& newCenters);
        void setKernelTrainingData(const Eigen::MatrixXd& newTrainingData);

        void addNewKernelData(const Eigen::VectorXd& newCenters, const Eigen::VectorXd& newTrainingData);
        void addNewKernelData(const Eigen::VectorXd& newCenters, const Eigen::MatrixXd& newTrainingData);
        void addNewKernelData(const Eigen::MatrixXd& newCenters, const Eigen::VectorXd& newTrainingData);
        void addNewKernelData(const Eigen::MatrixXd& newCenters, const Eigen::MatrixXd& newTrainingData);

        void removeKernelData(const int startIndex, const int nKernToRemove=1);
        void removeRecentlyAddedKernelData();

        void setMaxCovariance(const double maxCovScalar);
        void setMaxCovariance(const Eigen::VectorXd& maxCovVector);

        void setCovarianceMatrix(const Eigen::VectorXd& userSigmaVec);
        void setCovarianceMatrix(const Eigen::MatrixXd& userSigmaMat);

        void calculateMeanAndVariance(const Eigen::VectorXd& inputVector, Eigen::MatrixXd& mean, Eigen::MatrixXd& variance);
        void calculateMeanAndVariance(const Eigen::MatrixXd& inputVectors, Eigen::MatrixXd& means, Eigen::MatrixXd& variances);
        void calculateMean(const Eigen::VectorXd& inputVector, Eigen::VectorXd& mean);
        void calculateMean(const Eigen::MatrixXd& inputVectors, Eigen::MatrixXd& means);
        void calculateVariance(const Eigen::VectorXd& inputVector, Eigen::VectorXd& variance);
        void calculateVariance(const Eigen::MatrixXd& inputVectors, Eigen::MatrixXd& variances);


        void writeDataToFile(std::string directoryPath="./", const bool overwrite=false);
        void writeOutputToFile(std::string directoryPath="./", const bool overwrite=false);

        Eigen::MatrixXd getWeights(){return kernelWeights;}



    protected:

        // kernelFunction* kernelFuncPtr; /*!< Pointer to kernel function base class.*/
        squaredExponential* kernelFuncPtr; /*!< Pointer to squared exponential kernel function class.*/

        Eigen::VectorXd maximumCovariance;

        int outputDimension;

        Eigen::MatrixXd designMatrix;
        Eigen::MatrixXd designMatrixInv;
        Eigen::MatrixXd kernelCenters;
        Eigen::MatrixXd kernelWeights;
        Eigen::MatrixXd kernelTrainingData;

        int newKernelDataStartIndex;
        int newKernelDataSize;

    private:


    };
} // End of namespace smlt

#endif
