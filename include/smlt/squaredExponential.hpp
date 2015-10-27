#ifndef SQUAREDEXPONENTIAL_HPP
#define SQUAREDEXPONENTIAL_HPP

#include <iostream>
#include <fstream>
#include <stdlib.h>

#include "Eigen/Dense"

#include "smlt/smltUtilities.hpp"
#include "smlt/kernelFunction.hpp"


#include <ctime>



namespace smlt
{
    class squaredExponential : public kernelFunction
    {
    public:
        /*! Function Declarations */
        squaredExponential();
        squaredExponential(const Eigen::MatrixXd& centers);

        void setMaxCovariance(const double maxCovScalar=1.0);

        bool setCovarianceMatrix(const Eigen::VectorXd& userSigmaVec);
        bool setCovarianceMatrix(const Eigen::MatrixXd& userSigmaMat);

        Eigen::MatrixXd getCovarianceMatrix();

        void writeDataToFile(std::string directoryPath="./", const bool overwrite=false);
        void writeOutputToFile(std::string directoryPath="./", const bool overwrite=false);

        /*! Variables */



    protected:
        /*! Function Declarations */
        virtual void doInit();

        virtual void doEvaluate(const Eigen::VectorXd& inputVector, Eigen::MatrixXd& output);
        virtual void doEvaluate(const Eigen::MatrixXd& inputVectors, Eigen::MatrixXd& output);


        /*! Variables */
        Eigen::VectorXd sigmaVec;       /*!< A vector of variances for each dimension of the kernels */
        Eigen::MatrixXd sigmaMat;       /*!< The covariance matrix. */
        Eigen::MatrixXd sigmaMatInv;    /*!< The inverse of the covariance matrix. Precalculated to save time. */


        double maximumCovariance;       /*!< Maximum covariance of the kernel when exponential term goes to 1.0. */
    };


} // End of namespace smlt
#endif
