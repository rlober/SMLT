#include "smlt/bayesianOptimization.hpp"



using namespace smlt;

bayesianOptimization::bayesianOptimization(const bopt_Parameters optParams) :
optParameters(optParams)
{
    if (optParameters.logData) {
        createDataLog();
    }
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

    numberOfIterations++;

    int minIndex;
    minimizeAcquistionFunction(minIndex);

    currentSolution.currentMinCost = currentCostMeans(0,minIndex);
    currentSolution.currentConfidence = (1. - currentCostVariances(0,minIndex))*100.;

    currentSolution.optimalParameters = searchSpace.col(minIndex);
    currentSolution.nIter = numberOfIterations;

    bool isConfident = (currentSolution.currentConfidence>= optParameters.minConfidence);
    bool isMaxIter = (currentSolution.nIter >= optParameters.maxIter);
    currentSolution.optimumFound = isConfident || isMaxIter;

    if (optParameters.logData) {
        logOptimizationData();
    }

    if (!optParameters.silenceOutput)
    {
        if (currentSolution.optimumFound) {
            if (isConfident) {
                std::cout << "\n================================================================================\n";
                std::cout << "\tOptimum found with high confidence in "<< currentSolution.nIter <<" iterations!";
                std::cout << "\n================================================================================\n";
                std::cout << "\tOptimum = " << currentSolution.currentMinCost << std::endl;
                std::cout << "\tConfidence = " << currentSolution.currentConfidence  << "%"<< std::endl;
                std::cout << "\tOptimal parameters = " <<  currentSolution.optimalParameters.transpose() << std::endl;
                std::cout << "\n================================================================================" << std::endl;
            }
            if (isMaxIter) {
                std::cout << "\n================================================================================\n";
                std::cout << "\tWARNING: Maximum number of iterations ("<< optParameters.maxIter <<") exceeded.";
                std::cout << "\n================================================================================\n";
                std::cout << "\tCurrent optimum = " << currentSolution.currentMinCost << std::endl;
                std::cout << "\tConfidence = " << currentSolution.currentConfidence  << "%"<< std::endl;
                std::cout << "\tCurrent optimal parameters = " <<  currentSolution.optimalParameters.transpose() << std::endl;
                std::cout << "\n================================================================================" << std::endl;
            }
        }
        else{
            std::cout << "\n========================================\n";
            std::cout << "Optimization iteration: " << numberOfIterations << std::endl;
            std::cout << "-> Current minimum = " << currentSolution.currentMinCost << std::endl;
            std::cout << "-> Confidence = " << currentSolution.currentConfidence  << "%"<< std::endl;
            std::cout << "-> Optimal test parameters = " <<  currentSolution.optimalParameters.transpose() << std::endl;
            std::cout << "========================================\n" << std::endl;
        }
    }
    return currentSolution;
}

void bayesianOptimization::minimizeAcquistionFunction(int& optimalIndex)
{
    tau = tauFunction(numberOfIterations);
    LCB = currentCostMeans - sqrt(tau)*currentCostVariances;

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

void bayesianOptimization::createDataLog()
{

    optLogPath = optParameters.dataLogDir + "/optimization_log-" + currentDateTime() +"/";
    checkAndCreateDirectory(optLogPath);

    std::ofstream pathFile;
    pathFile.open((optParameters.dataLogDir+"/latestLogPath.txt").c_str());
    pathFile << optLogPath << std::endl;
    pathFile.close();
}

void bayesianOptimization::logOptimizationData()
{
    std::ostringstream it_stream;
    it_stream << numberOfIterations;
    std::string currentIterDir = optLogPath + "/" + it_stream.str()+ "/";
    checkAndCreateDirectory(currentIterDir);

    std::ofstream optParamsFile, optimumFoundFile, currentMinCostFile, currentConfidenceFile, optimalParametersFile, currentCostMeansFile, currentCostVariancesFile, LCBFile;



    optParamsFile.open((currentIterDir+"optParams.txt").c_str());
    optimumFoundFile.open((currentIterDir+"optimumFound.txt").c_str());
    currentMinCostFile.open((currentIterDir+"currentMinCost.txt").c_str());
    currentConfidenceFile.open((currentIterDir+"currentConfidence.txt").c_str());
    optimalParametersFile.open((currentIterDir+"optimalParameters.txt").c_str());
    currentCostMeansFile.open((currentIterDir+"currentCostMeans.txt").c_str());
    currentCostVariancesFile.open((currentIterDir+"currentCostVariances.txt").c_str());
    LCBFile.open((currentIterDir+"LCB.txt").c_str());

    optParamsFile               <<  optParameters;
    optimumFoundFile            <<  currentSolution.optimumFound;
    currentMinCostFile          <<  currentSolution.currentMinCost;
    currentConfidenceFile       <<  currentSolution.currentConfidence;
    optimalParametersFile       <<  currentSolution.optimalParameters;
    currentCostMeansFile        <<  currentCostMeans;
    currentCostVariancesFile    <<  currentCostVariances;
    LCBFile                     <<  LCB;


    optParamsFile.close();
    optimumFoundFile.close();
    currentMinCostFile.close();
    currentConfidenceFile.close();
    optimalParametersFile.close();
    currentCostMeansFile.close();
    currentCostVariancesFile.close();
    LCBFile.close();


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
