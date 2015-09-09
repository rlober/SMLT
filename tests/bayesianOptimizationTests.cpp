#include <iostream>
#include "Eigen/Dense"
#include "smlt/bayesianOptimization.hpp"

using namespace std;
using namespace smlt;

Eigen::VectorXd costFunction(const Eigen::VectorXd& input)
{
    Eigen::VectorXd output = input.array().square().sin();

    return output;
}


int main(int argc, char const *argv[])
{
    Eigen::VectorXd centerData = Eigen::VectorXd::Random(3);

    Eigen::VectorXd costData = costFunction(centerData);


    bopt_Parameters bopt_params = bopt_Parameters();

    bayesianOptimization bopt_solver = bayesianOptimization(bopt_params);

    bopt_Solution bopt_solution;

    bopt_solution = bopt_solver.initialize(centerData, costData);

    while (!bopt_solution.optimumFound) {

        //evalutate new parameters:
        Eigen::VectorXd newCost = costFunction(bopt_solution.optimalParameters);


        bopt_solution = bopt_solver.update(bopt_solution.optimalParameters, newCost);

    }


    return 0;
}
