#ifndef BAYESIANOPTIMIZATION_HPP
#define BAYESIANOPTIMIZATION_HPP

#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <math.h>

#include "Eigen/Dense"

#include "smlt/smltUtilities.hpp"
#include "smlt/gaussianProcess.hpp"

namespace smlt
{
    struct bopt_Solution
    {
        bool optimumFound; // Whether or not an optimum has been found through either minConfidence or maxIter
        double currentMinCost;  // The current minimum cost.
        double currentConfidence; // The current confidence in the minimum cost found.
        int nIter; // The number of optimization iterations so far

        Eigen::VectorXd optimalParameters; // The best parameters to use in the cost function.
    };

    struct bopt_Parameters
    {
        bopt_Parameters() // Set optimization parameters to their default values.
        {
            logData = true;
            silenceOutput = false;
            minConfidence = 99.99;
            maxIter = 50;
        }
        bool logData;
        bool silenceOutput; // no cout statements
        double minConfidence;
        int maxIter;
    };

    class bayesianOptimization
    {
    public:
        bayesianOptimization(const bopt_Parameters optParams);

        bopt_Solution initialize(const Eigen::VectorXd& centerData, const Eigen::VectorXd& costData);
        bopt_Solution initialize(const Eigen::VectorXd& centerData, const Eigen::MatrixXd& costData);
        bopt_Solution initialize(const Eigen::MatrixXd& centerData, const Eigen::VectorXd& costData);

        bopt_Solution initialize(const Eigen::MatrixXd& centerData, const Eigen::MatrixXd& costData);


        bopt_Solution update(const Eigen::VectorXd& newCenters, const Eigen::VectorXd& newCosts);
        bopt_Solution update(const Eigen::VectorXd& newCenters, const Eigen::MatrixXd& newCosts);
        bopt_Solution update(const Eigen::MatrixXd& newCenters, const Eigen::VectorXd& newCosts);

        bopt_Solution update(const Eigen::MatrixXd& newCenters, const Eigen::MatrixXd& newCosts);


        bopt_Solution solve();


        void setLoggerStatus(const bool logOptData=true);

        void setSearchSpace(const Eigen::MatrixXd& newSearchSpace);
        void recalculateSearchSpace();


    protected:
        void addNewDataToGP(const Eigen::MatrixXd& newCenters, const Eigen::MatrixXd& newCosts);
        void updateGaussianProcess();
        void minimizeAcquistionFunction(int& optimalIndex);
        double tauFunction(const int t);



        gaussianProcess* costGP;        /*!< The gaussian process estimating the cost function. */
        int numberOfIterations;         /*!< The current number of iterations of the optimization.*/
        /*!< The maximum number of iterations before the optimization stops. */
        /*!< The minimum confidence needed to consider a candidate solution an optimum. */

        Eigen::MatrixXd searchSpace;    /*!< The hyperrectangular search space of the optimization. */

        bopt_Parameters optParameters;

        Eigen::MatrixXd gpCenters;
        Eigen::MatrixXd gpTrainingData;

        Eigen::MatrixXd currentCostMeans;
        Eigen::MatrixXd currentCostVariances;

    private:

    };
} // End of namespace smlt

#endif
