#include "smlt/bayesianOptimization.hpp"



using namespace smlt;

bayesianOptimization::bayesianOptimization(const bopt_Parameters optParams) :
optParameters(optParams)
{

}

bopt_Solution bayesianOptimization::initialize(const Eigen::VectorXd& centerData, const Eigen::VectorXd& costData)
{
    return initialize(Eigen::MatrixXd(centerData.transpose()), Eigen::MatrixXd(costData.transpose()));
}

bopt_Solution bayesianOptimization::initialize(const Eigen::VectorXd& centerData, const Eigen::MatrixXd& costData)
{
    return initialize(Eigen::MatrixXd(centerData.transpose()), costData);
}

bopt_Solution bayesianOptimization::initialize(const Eigen::MatrixXd& centerData, const Eigen::VectorXd& costData)
{
    return initialize(centerData, Eigen::MatrixXd(costData.transpose()));
}

bopt_Solution bayesianOptimization::initialize(const Eigen::MatrixXd& centerData, const Eigen::MatrixXd& costData)
{

    gpCenters = centerData;
    gpTrainingData = costData;
    numberOfIterations = 0;

    Eigen::VectorXd mins = gpCenters.rowwise().minCoeff();
    Eigen::VectorXd maxs = gpCenters.rowwise().maxCoeff();

    int nSteps = 50;

    searchSpace = discretizeSearchSpace(mins, maxs, nSteps);

    costGP = new gaussianProcess();

    updateGaussianProcess();


    return solve();
}




bopt_Solution bayesianOptimization::update(const Eigen::VectorXd& newCenters, const Eigen::VectorXd& newCosts)
{
    return update(Eigen::MatrixXd(newCenters.transpose()), Eigen::MatrixXd(newCosts.transpose()));
}

bopt_Solution bayesianOptimization::update(const Eigen::VectorXd& newCenters, const Eigen::MatrixXd& newCosts)
{
    return update(Eigen::MatrixXd(newCenters.transpose()), newCosts);
}

bopt_Solution bayesianOptimization::update(const Eigen::MatrixXd& newCenters, const Eigen::VectorXd& newCosts)
{
    return update(newCenters, Eigen::MatrixXd(newCosts.transpose()));
}


bopt_Solution bayesianOptimization::update(const Eigen::MatrixXd& newCenters, const Eigen::MatrixXd& newCosts)
{

    addNewDataToGP(newCenters, newCosts);
    updateGaussianProcess();

    return solve();
}

bopt_Solution bayesianOptimization::solve()
{
    bopt_Solution solution;
    numberOfIterations++;

    int minIndex;
    minimizeAcquistionFunction(minIndex);

    solution.currentMinCost = currentCostMeans(0,minIndex);
    solution.currentConfidence = (1. - currentCostVariances(0,minIndex))*100.;

    solution.optimalParameters = searchSpace.col(minIndex);
    solution.nIter = numberOfIterations;

    bool isConfident = (solution.currentConfidence>= optParameters.minConfidence);
    bool isMaxIter = (solution.nIter >= optParameters.maxIter);
    solution.optimumFound = isConfident || isMaxIter;

    if (!optParameters.silenceOutput)
    {
        if (solution.optimumFound) {
            if (isConfident) {
                std::cout << "\n================================================================================\n";
                std::cout << "\tOptimum found with high confidence in "<< solution.nIter <<" iterations!";
                std::cout << "\n================================================================================\n";
                std::cout << "\tOptimum = " << solution.currentMinCost << std::endl;
                std::cout << "\tConfidence = " << solution.currentConfidence  << "%"<< std::endl;
                std::cout << "\tOptimal parameters = " <<  solution.optimalParameters.transpose() << std::endl;
                std::cout << "\n================================================================================" << std::endl;
            }
            if (isMaxIter) {
                std::cout << "\n================================================================================\n";
                std::cout << "\tWARNING: Maximum number of iterations ("<< optParameters.maxIter <<") exceeded.";
                std::cout << "\n================================================================================\n";
                std::cout << "\tCurrent optimum = " << solution.currentMinCost << std::endl;
                std::cout << "\tConfidence = " << solution.currentConfidence  << "%"<< std::endl;
                std::cout << "\tCurrent optimal parameters = " <<  solution.optimalParameters.transpose() << std::endl;
                std::cout << "\n================================================================================" << std::endl;
            }
        }
        else{
            std::cout << "\n========================================\n";
            std::cout << "Optimization iteration: " << numberOfIterations << std::endl;
            std::cout << "-> Current minimum = " << solution.currentMinCost << std::endl;
            std::cout << "-> Confidence = " << solution.currentConfidence  << "%"<< std::endl;
            std::cout << "-> Optimal test parameters = " <<  solution.optimalParameters.transpose() << std::endl;
            std::cout << "========================================\n" << std::endl;
        }
    }
    return solution;
}

void bayesianOptimization::minimizeAcquistionFunction(int& optimalIndex)
{
    double tau = tauFunction(numberOfIterations);
    Eigen::MatrixXd LCB = currentCostMeans - sqrt(tau)*currentCostVariances;
    Eigen::MatrixXd::Index dummy, idx;
    LCB.minCoeff(&dummy, &idx);
    optimalIndex = idx;
}

double bayesianOptimization::tauFunction(const int t)
{
    double d = 1.0; // determines the upper asymptote of the function.
    double delta = 1.0; // Essentially does nothing. basically Adjusts the starting value when t = 0

    return 2.0*log(pow((double)t, ((d/2.0)+2.0)) + (pow(M_PI, 2.0) / 3.0*delta) );
}


void bayesianOptimization::addNewDataToGP(const Eigen::MatrixXd& newCenters, const Eigen::MatrixXd& newCosts)
{
    int oldCols = gpCenters.cols();
    int extraCols = newCenters.cols();
    Eigen::MatrixXd tmpCenters(gpCenters.rows(), oldCols+extraCols);
    tmpCenters << gpCenters, newCenters;

    gpCenters.resize(tmpCenters.rows(), tmpCenters.cols());
    gpCenters = tmpCenters;

    Eigen::MatrixXd tmpTrain(gpTrainingData.rows(), oldCols+extraCols);
    tmpTrain << gpTrainingData, newCosts;

    gpTrainingData.resize(tmpTrain.rows(), tmpTrain.cols());
    gpTrainingData = tmpTrain;
}

void bayesianOptimization::updateGaussianProcess()
{
    costGP->setKernelCenters(gpCenters);

    costGP->setKernelTrainingData(gpTrainingData);

    costGP->setCovarianceMatrix(calculateCovariance(gpCenters, true));

    costGP->setMaxCovariance(getVariance(Eigen::MatrixXd(gpTrainingData.transpose())));

    costGP->calculateParameters();

    costGP->getMeanAndVariance(searchSpace, currentCostMeans, currentCostVariances);

}

void bayesianOptimization::setLoggerStatus(const bool logOptData)
{

}

void bayesianOptimization::setSearchSpace(const Eigen::MatrixXd& newSearchSpace)
{

}

void bayesianOptimization::recalculateSearchSpace()
{

}
