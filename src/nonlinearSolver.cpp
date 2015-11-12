#include "smlt/nonlinearSolver.hpp"



using namespace smlt;

nonlinearSolver::nonlinearSolver(const optParameters _optParams) :
optParams(_optParams)
{
}


optSolution nonlinearSolver::initialize(const Eigen::VectorXd& optVariables, const Eigen::VectorXd& costs)
{
    return initialize(Eigen::MatrixXd(optVariables.transpose()), Eigen::MatrixXd(costs.transpose()));
}

optSolution nonlinearSolver::initialize(const Eigen::VectorXd& optVariables, const Eigen::MatrixXd& costs)
{
    return initialize(Eigen::MatrixXd(optVariables.transpose()), costs);
}

optSolution nonlinearSolver::initialize(const Eigen::MatrixXd& optVariables, const Eigen::VectorXd& costs)
{
    return initialize(optVariables, Eigen::MatrixXd(costs.transpose()));
}

optSolution nonlinearSolver::initialize(const Eigen::MatrixXd& optVariables, const Eigen::MatrixXd& costs)
{
    return doInit(optVariables, costs);
}


optSolution nonlinearSolver::update(Eigen::VectorXd& newOptVariables, Eigen::VectorXd& newCosts)
{
    if (optVars.rows()!=newOptVariables.rows())
    {
        newOptVariables.transposeInPlace();
    }
    return update(Eigen::MatrixXd(newOptVariables), Eigen::MatrixXd(newCosts.transpose()));
}

optSolution nonlinearSolver::update(Eigen::VectorXd& newOptVariables, Eigen::MatrixXd& newCosts)
{
    if (optVars.rows()!=newOptVariables.rows())
    {
        newOptVariables.transposeInPlace();
    }
    return update(Eigen::MatrixXd(newOptVariables), newCosts);
}

optSolution nonlinearSolver::update(Eigen::MatrixXd& newOptVariables, Eigen::VectorXd& newCosts)
{
    return update(newOptVariables, Eigen::MatrixXd(newCosts.transpose()));
}


optSolution nonlinearSolver::update(const Eigen::MatrixXd& newOptVariables, const Eigen::MatrixXd& newCosts)
{
    return doUpdate(newOptVariables, newCosts);
}
