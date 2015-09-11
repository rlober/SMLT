#include <iostream>
#include "Eigen/Dense"
#include "smlt/bayesianOptimization.hpp"
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


    bopt_Parameters bopt_params = bopt_Parameters();

    bopt_params.dataLogDir = "/home/ryan/Code/smlt/tmp3/";

    bayesianOptimization bopt_solver = bayesianOptimization(bopt_params);

    bopt_Solution bopt_solution;

    bopt_solution = bopt_solver.initialize(centerData, costData);

    while (!bopt_solution.optimumFound) {

        //evalutate new parameters:
        Eigen::VectorXd newCost = costFunction(bopt_solution.optimalParameters);


        bopt_solution = bopt_solver.update(bopt_solution.optimalParameters, newCost);

    }


    std::system(("python ./scripts/plotBayesianOptimization.py "+ bopt_params.dataLogDir).c_str());


    return 0;
}
