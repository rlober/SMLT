#ifndef BAYESIANOPTIMIZATION_HPP
#define BAYESIANOPTIMIZATION_HPP

#include <iostream>
#include <fstream>
#include <sstream>
#include <ostream>
#include <stdlib.h>
#include <math.h>

#include "Eigen/Dense"

#include "smlt/smltUtilities.hpp"
#include "smlt/gaussianProcess.hpp"
#include "smlt/nonlinearSolver.hpp"

namespace smlt
{

    class bayesianOptimization : public nonlinearSolver
    {
    public:
        bayesianOptimization(const optParameters optParams);

        void setLoggerStatus(const bool logOptData=true);

        void setSearchSpace(const Eigen::MatrixXd& newSearchSpace);
        void recalculateSearchSpace();



    protected:

        virtual optSolution doInit(const Eigen::MatrixXd& optVariables, const Eigen::MatrixXd& costs);

        virtual optSolution doUpdate(const Eigen::MatrixXd& newOptVariables, const Eigen::MatrixXd& newCosts);

        optSolution solve();



        void updateGaussianProcess();
        void minimizeAcquistionFunction(int& optimalIndex);
        double tauFunction(const int t);

        void createDataLog();
        void logOptimizationData();

        std::string optLogPath;
        double tau;
        Eigen::MatrixXd LCB;
        optSolution currentSolution;

        gaussianProcess* costGP;        /*!< The gaussian process estimating the cost function. */
        int numberOfIterations;         /*!< The current number of iterations of the optimization.*/
        /*!< The maximum number of iterations before the optimization stops. */
        /*!< The minimum confidence needed to consider a candidate solution an optimum. */

        Eigen::MatrixXd searchSpace;    /*!< The hyperrectangular search space of the optimization. */
        Eigen::MatrixXd searchSpaceBounds;




        Eigen::MatrixXd currentCostMeans;
        Eigen::MatrixXd currentCostVariances;

        Eigen::VectorXd normalizationRanges, normalizationMins;

        Eigen::MatrixXd costCovariance;
        Eigen::VectorXd costMaxCovariance;


        bool covarianceSetByUser;

        double optVarCostScalingFactor;

    private:

    };
} // End of namespace smlt

#endif
