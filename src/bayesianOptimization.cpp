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

    Eigen::VectorXd minVec;
    Eigen::VectorXd maxVec;

    if ((optParameters.searchSpaceMinBound.size()!=0) && (optParameters.searchSpaceMaxBound.size()!=0)) {
        minVec = optParameters.searchSpaceMinBound;
        maxVec = optParameters.searchSpaceMaxBound;
    }
    else{
        if (gpCenters.cols()>1){
            minVec = gpCenters.rowwise().minCoeff();
            maxVec = gpCenters.rowwise().maxCoeff();
        }else{
            smltError("You need to specify a search grid, or provide at least two data points.");
        }
    }


    Eigen::VectorXi nSteps;

    if ((optParameters.gridSpacing.size()==0) && (optParameters.gridSteps.size()==0)){
        nSteps = Eigen::VectorXi::Constant(minVec.rows(), 10);
    }
    if (optParameters.gridSpacing.size()>0){
        nSteps = ((optParameters.searchSpaceMaxBound - optParameters.searchSpaceMinBound).array() / optParameters.gridSpacing.array()).cast <int> ();

    }
    if (optParameters.gridSteps.size()>0) {
        nSteps = optParameters.gridSteps;
    }

    if (optParameters.normalize) {
        normalizationMins = optParameters.searchSpaceMinBound;
        normalizationRanges = optParameters.searchSpaceMaxBound - normalizationMins;
        int nOptVars = optParameters.searchSpaceMaxBound.rows();
        minVec = Eigen::VectorXd::Zero(nOptVars);
        maxVec = Eigen::VectorXd::Ones(nOptVars);
        gpCenters = (gpCenters.colwise() - normalizationMins).array() / normalizationRanges.replicate(1,gpCenters.cols()).array();
    }

    searchSpaceBounds.resize(2, minVec.rows());
    searchSpaceBounds << minVec.transpose(), maxVec.transpose();


    discretizeSearchSpace(minVec, maxVec, nSteps, searchSpace);






    if (optParameters.costCovariance.rows()>0 && optParameters.costMaxCovariance.rows()>0) {


        if (optParameters.normalize) {
            costCovariance = ((optParameters.costCovariance.diagonal() - normalizationMins).array() / normalizationRanges.array()).matrix().asDiagonal();
            costMaxCovariance = optParameters.costMaxCovariance;
        }
        else {
            costCovariance = optParameters.costCovariance;
            costMaxCovariance = optParameters.costMaxCovariance;
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




bopt_Solution bayesianOptimization::update(Eigen::VectorXd& newCenters, Eigen::VectorXd& newCosts)
{
    if (gpCenters.rows()!=newCenters.rows())
    {
        newCenters.transposeInPlace();
    }
    return update(Eigen::MatrixXd(newCenters), Eigen::MatrixXd(newCosts.transpose()));
}

bopt_Solution bayesianOptimization::update(Eigen::VectorXd& newCenters, Eigen::MatrixXd& newCosts)
{
    if (gpCenters.rows()!=newCenters.rows())
    {
        newCenters.transposeInPlace();
    }
    return update(Eigen::MatrixXd(newCenters), newCosts);
}

bopt_Solution bayesianOptimization::update(Eigen::MatrixXd& newCenters, Eigen::VectorXd& newCosts)
{
    return update(newCenters, Eigen::MatrixXd(newCosts.transpose()));
}


bopt_Solution bayesianOptimization::update(const Eigen::MatrixXd& newCenters, const Eigen::MatrixXd& newCosts)
{
    Eigen::MatrixXd _newCenters = newCenters;
    if (optParameters.normalize) {
        _newCenters = (_newCenters - normalizationMins).array() / normalizationRanges.array();
    }
    gpCenters = hStack(gpCenters, _newCenters);
    gpTrainingData = hStack(gpTrainingData, newCosts);



    updateGaussianProcess();

    return solve();
}

bopt_Solution bayesianOptimization::solve()
{

    numberOfIterations++;


    minimizeAcquistionFunction(currentSolution.minIndex);


    currentSolution.currentMinCost = currentCostMeans(0,currentSolution.minIndex);
    currentSolution.currentConfidence = (1. - currentCostVariances(0,currentSolution.minIndex))*100.;

    currentSolution.optimalParameters = searchSpace.col(currentSolution.minIndex);




    currentSolution.nIter = numberOfIterations;

    bool isConfident = (currentSolution.currentConfidence>= optParameters.minConfidence);
    bool isMaxIter = (currentSolution.nIter >= optParameters.maxIter);
    currentSolution.optimumFound = isConfident || isMaxIter;

    if (optParameters.logData) {
        logOptimizationData();
    }

    if (optParameters.normalize) {
        currentSolution.optimalParameters = (currentSolution.optimalParameters.array() * normalizationRanges.array()).matrix() + normalizationMins;
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

void bayesianOptimization::updateGaussianProcess()
{
    costGP->setKernelCenters(gpCenters);
    costGP->setKernelTrainingData(gpTrainingData);


    if (covarianceSetByUser)
    {
        bool test1 = costCovariance.rows() == gpCenters.rows();
        bool test2 = costMaxCovariance.rows() == 1;
        if (test1 && test2) {
            costGP->setCovarianceMatrix(costCovariance);
            costGP->setMaxCovariance(costMaxCovariance);
        }else{
            if (!test1) {
                smltError("Provided kernel covariance matrix dimension (" << costCovariance.rows() <<"x"<< costCovariance.cols() << ") does not match kernel dimension ("<<  gpCenters.rows() << ").");
            }
            if (!test2) {
                smltError("Provided kernel maximum covariance vector dimension (" << costMaxCovariance.rows()<< ") is greater than 1.");
            }
        }
    }
    else
    {
        Eigen::MatrixXd covMat = calculateCovariance(gpCenters, true).array().abs()*10.0;
        std::cout << "\ncovMat\n" << covMat << std::endl;
        costGP->setCovarianceMatrix(covMat);
        if (gpTrainingData.rows()==1) {
            costGP->setMaxCovariance(1.0);

        }else{
            costGP->setMaxCovariance(getVariance(Eigen::MatrixXd(gpTrainingData.transpose())));

        }
    }


    costGP->calculateParameters();
    costGP->calculateMeanAndVariance(searchSpace, currentCostMeans, currentCostVariances);
}

void bayesianOptimization::createDataLog()
{

    optLogPath = optParameters.dataLogDir + "/optimization_log-" + currentDateTime() +"/";
    checkAndCreateDirectory(optLogPath);

    std::ofstream pathFile;
    pathFile.open((optParameters.dataLogDir+"/latestLogPath.txt").c_str());
    pathFile << optLogPath;
    pathFile.close();

    std::ofstream optParamsFile;
    optParamsFile.open((optLogPath+"/optParams.txt").c_str());
    optParamsFile << optParameters;
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


    gpParamsFile                <<  gpCenters;
    gpWeightsFile               <<  costGP->getWeights();
    gpCostsFile                 <<  gpTrainingData;

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
