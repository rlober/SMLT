#include "smlt/bayesianOptimization.hpp"



using namespace smlt;

bayesianOptimization::bayesianOptimization(const optParameters _optParams) :
nonlinearSolver(_optParams)
{
    if (optParams.logData) {
        createDataLog();
    }
}

optSolution bayesianOptimization::doInit(const Eigen::MatrixXd& centerData, const Eigen::MatrixXd& costData)
{

    optVars = centerData;
    optVarCosts = costData;
    numberOfIterations = 0;

    Eigen::VectorXd minVec;
    Eigen::VectorXd maxVec;

    if ((optParams.searchSpaceMinBound.size()!=0) && (optParams.searchSpaceMaxBound.size()!=0)) {
        minVec = optParams.searchSpaceMinBound;
        maxVec = optParams.searchSpaceMaxBound;
    }
    else{
        if (optVars.cols()>1){
            minVec = optVars.rowwise().minCoeff();
            maxVec = optVars.rowwise().maxCoeff();
        }else{
            smltError("You need to specify a search grid, or provide at least two data points.");
        }
    }


    Eigen::VectorXi nSteps;

    if ((optParams.gridSpacing.size()==0) && (optParams.gridSteps.size()==0)){
        nSteps = Eigen::VectorXi::Constant(minVec.rows(), 10);
    }
    if (optParams.gridSpacing.size()>0){
        nSteps = ((optParams.searchSpaceMaxBound - optParams.searchSpaceMinBound).array() / optParams.gridSpacing.array()).cast <int> ();

    }
    if (optParams.gridSteps.size()>0) {
        nSteps = optParams.gridSteps;
    }

    if (optParams.normalize) {
        normalizationMins = optParams.searchSpaceMinBound;
        normalizationRanges = optParams.searchSpaceMaxBound - normalizationMins;
        int nOptVars = optParams.searchSpaceMaxBound.rows();
        minVec = Eigen::VectorXd::Zero(nOptVars);
        maxVec = Eigen::VectorXd::Ones(nOptVars);
        optVars = (optVars.colwise() - normalizationMins).array() / normalizationRanges.replicate(1,optVars.cols()).array();
    }

    searchSpaceBounds.resize(2, minVec.rows());
    searchSpaceBounds << minVec.transpose(), maxVec.transpose();


    discretizeSearchSpace(minVec, maxVec, nSteps, searchSpace);






    if (optParams.costCovariance.rows()>0 && optParams.costMaxCovariance.rows()>0) {


        if (optParams.normalize) {
            costCovariance = ((optParams.costCovariance.diagonal() - normalizationMins).array() / normalizationRanges.array()).matrix().asDiagonal();
            costMaxCovariance = optParams.costMaxCovariance;
        }
        else {
            costCovariance = optParams.costCovariance;
            costMaxCovariance = optParams.costMaxCovariance;
        }

        covarianceSetByUser=true;
    }else{
        covarianceSetByUser=false;
    }



    costGP = new gaussianProcess();

    updateGaussianProcess();

    covarianceSetByUser = false;

    return solve();
}


optSolution bayesianOptimization::doUpdate(const Eigen::MatrixXd& newCenters, const Eigen::MatrixXd& newCosts)
{
    Eigen::MatrixXd _newCenters = newCenters;
    if (optParams.normalize) {
        _newCenters = (_newCenters - normalizationMins).array() / normalizationRanges.array();
    }
    optVars = hStack(optVars, _newCenters);
    optVarCosts = hStack(optVarCosts, newCosts);



    updateGaussianProcess();

    return solve();
}

optSolution bayesianOptimization::solve()
{

    numberOfIterations++;


    minimizeAcquistionFunction(currentSolution.minIndex);


    currentSolution.currentMinCost = currentCostMeans(0,currentSolution.minIndex);
    currentSolution.currentConfidence = (1. - currentCostVariances(0,currentSolution.minIndex))*100.;

    currentSolution.optimalParameters = searchSpace.col(currentSolution.minIndex);




    currentSolution.nIter = numberOfIterations;

    bool isConfident = (currentSolution.currentConfidence>= optParams.minConfidence);
    bool isMaxIter = (currentSolution.nIter >= optParams.maxIter);
    currentSolution.optimumFound = isConfident || isMaxIter;

    if (optParams.logData) {
        logOptimizationData();
    }

    if (optParams.normalize) {
        currentSolution.optimalParameters = (currentSolution.optimalParameters.array() * normalizationRanges.array()).matrix() + normalizationMins;
    }

    if (!optParams.silenceOutput)
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
                std::cout << "\tWARNING: Maximum number of iterations ("<< optParams.maxIter <<") exceeded.";
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
    // double ccmMin = currentCostMeans.minCoeff();
    // double ccmMax = currentCostMeans.maxCoeff();
    // double ccmRange = ccmMax - ccmMin;
    // LCB = (currentCostMeans.array() - ccmMin).matrix()/(ccmRange) - sqrt(tau)*currentCostVariances;

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

void bayesianOptimization::updateGaussianProcess()
{
    costGP->setKernelCenters(optVars);
    costGP->setKernelTrainingData(optVarCosts);
    /* An attempt to scale costs from 0-1; Doesn't work well.
    if (optParams.normalize) {

        Eigen::MatrixXd optVarCostsNormalized;

        if (optVarCosts.cols()==1) {
            optVarCostsNormalized = Eigen::MatrixXd::Ones(optVarCosts.rows(), optVarCosts.cols());
        }
        else{
            optVarCostsNormalized = (optVarCosts.colwise() - optVarCosts.rowwise().minCoeff()).array() / (optVarCosts.rowwise().maxCoeff() - optVarCosts.rowwise().minCoeff()).replicate(1,optVarCosts.cols()).array();
        }

        costGP->setKernelTrainingData(optVarCostsNormalized);
    }
    else {
        costGP->setKernelTrainingData(optVarCosts);
    }
    */


    if (covarianceSetByUser)
    {
        bool test1 = costCovariance.rows() == optVars.rows();
        bool test2 = costMaxCovariance.rows() == 1;
        if (test1 && test2) {
            costGP->setCovarianceMatrix(costCovariance);
            costGP->setMaxCovariance(costMaxCovariance);
        }else{
            if (!test1) {
                smltError("Provided kernel covariance matrix dimension (" << costCovariance.rows() <<"x"<< costCovariance.cols() << ") does not match kernel dimension ("<<  optVars.rows() << ").");
            }
            if (!test2) {
                smltError("Provided kernel maximum covariance vector dimension (" << costMaxCovariance.rows()<< ") is greater than 1.");
            }
        }
    }
    else
    {
        Eigen::MatrixXd covMat = calculateCovariance(optVars, true).array().abs();//*10.0;
        std::cout << "\ncovMat\n" << covMat << std::endl;
        costGP->setCovarianceMatrix(covMat);
        if (optVarCosts.rows()==1) {
            costGP->setMaxCovariance(1.0);

        }else{
            costGP->setMaxCovariance(getVariance(Eigen::MatrixXd(optVarCosts.transpose())));

        }
    }


    costGP->calculateParameters();
    costGP->calculateMeanAndVariance(searchSpace, currentCostMeans, currentCostVariances);
}

void bayesianOptimization::createDataLog()
{

    optLogPath = optParams.dataLogDir + "/optimization_log-" + currentDateTime() +"/";
    checkAndCreateDirectory(optLogPath);

    std::ofstream pathFile;
    pathFile.open((optParams.dataLogDir+"/latestLogPath.txt").c_str());
    pathFile << optLogPath;
    pathFile.close();

    std::ofstream optParamsFile;
    optParamsFile.open((optLogPath+"/optParams.txt").c_str());
    optParamsFile << optParams;
    optParamsFile.close();
}

void bayesianOptimization::logOptimizationData()
{
    std::ostringstream it_stream;
    it_stream << numberOfIterations;
    std::string currentIterDir = optLogPath + "/" + it_stream.str()+ "/";
    checkAndCreateDirectory(currentIterDir);

    std::ofstream gpParamsFile, gpWeightsFile, gpCostsFile, minIndexFile, optimumFoundFile, currentMinCostFile, currentConfidenceFile, optimalParametersFile, currentCostMeansFile, currentCostVariancesFile, LCBFile, searchSpaceFile, searchSpaceBoundsFile, tauFile;



    gpParamsFile.open((currentIterDir+"gpParams.txt").c_str());
    gpWeightsFile.open((currentIterDir+"gpWeights.txt").c_str());
    gpCostsFile.open((currentIterDir+"gpCosts.txt").c_str());

    minIndexFile.open((currentIterDir+"minIndex.txt").c_str());
    optimumFoundFile.open((currentIterDir+"optimumFound.txt").c_str());
    currentMinCostFile.open((currentIterDir+"currentMinCost.txt").c_str());
    currentConfidenceFile.open((currentIterDir+"currentConfidence.txt").c_str());
    optimalParametersFile.open((currentIterDir+"optimalParameters.txt").c_str());
    currentCostMeansFile.open((currentIterDir+"currentCostMeans.txt").c_str());
    currentCostVariancesFile.open((currentIterDir+"currentCostVariances.txt").c_str());
    LCBFile.open((currentIterDir+"LCB.txt").c_str());
    searchSpaceFile.open((currentIterDir+"searchSpace.txt").c_str());
    searchSpaceBoundsFile.open((currentIterDir+"searchSpaceBounds.txt").c_str());
    tauFile.open((currentIterDir+"tau.txt").c_str());


    gpParamsFile                <<  optVars;
    gpWeightsFile               <<  costGP->getWeights();
    gpCostsFile                 <<  optVarCosts;

    minIndexFile                <<  currentSolution.minIndex;
    optimumFoundFile            <<  currentSolution.optimumFound;
    currentMinCostFile          <<  currentSolution.currentMinCost;
    currentConfidenceFile       <<  currentSolution.currentConfidence;
    optimalParametersFile       <<  currentSolution.optimalParameters;
    currentCostMeansFile        <<  currentCostMeans;
    currentCostVariancesFile    <<  currentCostVariances;
    LCBFile                     <<  LCB;
    searchSpaceFile             <<  searchSpace;
    searchSpaceBoundsFile       <<  searchSpaceBounds;
    tauFile                     <<  tau;



    gpParamsFile.close();
    gpWeightsFile.close();
    gpCostsFile.close();

    minIndexFile.close();
    optimumFoundFile.close();
    currentMinCostFile.close();
    currentConfidenceFile.close();
    optimalParametersFile.close();
    currentCostMeansFile.close();
    currentCostVariancesFile.close();
    LCBFile.close();
    searchSpaceFile.close();
    searchSpaceBoundsFile.close();
    tauFile.close();


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
