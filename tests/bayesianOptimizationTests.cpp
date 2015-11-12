#include <iostream>
#include "Eigen/Dense"
#include "smlt/bayesianOptimization.hpp"
#include "smlt/nonlinearSolver.hpp"
#include <stdlib.h>



using namespace std;
using namespace smlt;

Eigen::VectorXd costFunction(const Eigen::VectorXd& input)
{
    // Eigen::VectorXd output = input.array().square().sin();
    Eigen::VectorXd output = input.array().sin()*3.33 + input.array().square()/10.0 + input.array().pow(3)/100.0;

    return output;
}


int main(int argc, char const *argv[])
{

    Eigen::VectorXd centerData(2);
    centerData << 1.0, 7.12;
    Eigen::VectorXd costData = costFunction(centerData);


    optParameters bopt_params = optParameters();

    bopt_params.searchSpaceMinBound = Eigen::VectorXd::Constant(1, 0.1);
    bopt_params.searchSpaceMaxBound = Eigen::VectorXd::Constant(1, 8.0);
    bopt_params.gridSteps = Eigen::VectorXi::Constant(1, 50);
    bopt_params.maxIter = 50;
    bopt_params.normalize = true;


    bopt_params.dataLogDir = "/home/ryan/Code/smlt/tmp3/";

    nonlinearSolver* bopt_solver = new bayesianOptimization(bopt_params);

    optSolution optSolution;

    optSolution = bopt_solver->initialize(centerData, costData);

    while (!optSolution.optimumFound) {

        //evalutate new parameters:
        Eigen::VectorXd newCost = costFunction(optSolution.optimalParameters);


        optSolution = bopt_solver->update(optSolution.optimalParameters, newCost);

    }


    std::system(("python ./scripts/plotBayesianOptimization.py "+ bopt_params.dataLogDir).c_str());


    return 0;
}
