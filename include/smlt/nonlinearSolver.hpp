#ifndef NONLINEARSOLVER_HPP
#define NONLINEARSOLVER_HPP

#include <iostream>
#include <fstream>
#include <sstream>
#include <ostream>
#include <stdlib.h>

#include "Eigen/Dense"


namespace smlt
{
    struct optSolution
    {
        bool optimumFound; // Whether or not an optimum has been found through either minConfidence or maxIter
        double currentMinCost;  // The current minimum cost.
        double currentConfidence; // The current confidence in the minimum cost found.
        int nIter; // The number of optimization iterations so far
        int minIndex; // The index of the minimum LCB value.

        Eigen::VectorXd optimalParameters; // The best parameters to use in the cost function.
    };

    struct optParameters
    {
        optParameters() // Set optimization parameters to their default values.
        {
            logData = true;
            dataLogDirPrefix = "./";
            dataLogDir = "";
            silenceOutput = false;
            minConfidence = 99.99;
            maxIter = 50;
            normalize = false;
        }
        bool logData;
        bool silenceOutput; // no cout statements
        double minConfidence;
        int maxIter;
        std::string dataLogDirPrefix;
        std::string dataLogDir;
        Eigen::VectorXd searchSpaceMinBound;
        Eigen::VectorXd searchSpaceMaxBound;
        Eigen::VectorXd gridSpacing;
        Eigen::VectorXi gridSteps;
        Eigen::MatrixXd costCovariance;
        Eigen::VectorXd costMaxCovariance;
        bool normalize;


        friend std::ostream& operator<<(std::ostream &out, const optParameters& params)
        {
            out << "logData = " << params.logData << std::endl;
            out << "dataLogDirPrefix = " << params.dataLogDirPrefix << std::endl;
            out << "dataLogDir = " << params.dataLogDir << std::endl;
            out << "silenceOutput = " << params.silenceOutput << std::endl;
            out << "minConfidence = " << params.minConfidence << std::endl;
            out << "maxIter = " << params.maxIter << std::endl;
            out << "searchSpaceMinBound = " << params.searchSpaceMinBound.transpose() << std::endl;
            out << "searchSpaceMaxBound = " << params.searchSpaceMaxBound.transpose() << std::endl;
            out << "gridSpacing = " << params.gridSpacing.transpose() << std::endl;
            out << "gridSteps = " << params.gridSteps.transpose() << std::endl;
            out << "normalize = " << params.normalize << std::endl;
            return out;
        }


    };

    class nonlinearSolver
    {
    public:
        nonlinearSolver(const optParameters optParams);

        optSolution initialize(const Eigen::VectorXd& optVariables, const Eigen::VectorXd& costs);
        optSolution initialize(const Eigen::VectorXd& optVariables, const Eigen::MatrixXd& costs);
        optSolution initialize(const Eigen::MatrixXd& optVariables, const Eigen::VectorXd& costs);
        optSolution initialize(const Eigen::MatrixXd& optVariables, const Eigen::MatrixXd& costs);


        optSolution update(Eigen::VectorXd& newOptVariables, Eigen::VectorXd& newCosts);
        optSolution update(Eigen::VectorXd& newOptVariables, Eigen::MatrixXd& newCosts);
        optSolution update(Eigen::MatrixXd& newOptVariables, Eigen::VectorXd& newCosts);
        optSolution update(const Eigen::MatrixXd& newOptVariables, const Eigen::MatrixXd& newCosts);



    protected:
        virtual optSolution doInit(const Eigen::MatrixXd& optVariables, const Eigen::MatrixXd& costs) = 0;
        virtual optSolution doUpdate(const Eigen::MatrixXd& newOptVariables, const Eigen::MatrixXd& newCosts) = 0;

        Eigen::MatrixXd optVars;
        Eigen::MatrixXd optVarCosts;
        optParameters optParams;

    private:

    };
} // End of namespace smlt

#endif
