#include "smlt/smltUtilities.hpp"

namespace smlt{

Eigen::VectorXd getVariance(const Eigen::MatrixXd& inputMat)
{
    Eigen::VectorXd result = Eigen::VectorXd::Zero(inputMat.cols());
    for (size_t j = 0; j < inputMat.cols(); j++) {
        result.row(j) << getVariance(Eigen::VectorXd(inputMat.col(j)));
    }

    return result;
}

double getVariance(const Eigen::VectorXd& inputVec)
{
    return (inputVec - Eigen::VectorXd::Constant(inputVec.size(), inputVec.mean())).array().square().sum() / (inputVec.size()-1);
}

// void removeRows(Eigen::VectorXd& vecToModify, const int startIndex, const int nRows)
// {
//     Eigen::MatrixXd tmpVec = Eigen::MatrixXd(vecToModify);
//     removeRows(tmpVec, startIndex, nRows);
//     vecToModify = Eigen::VectorXd(tmpVec);
// }
// void removeRows(Eigen::MatrixXd& matToModify, const int startIndex, const int nRows)
// {
//     matToModify.transposeInPlace();
//     removeCols(matToModify, startIndex, nRows);
//     matToModify.transposeInPlace();
// }

// void removeCols(Eigen::MatrixXd& matToModify, const int startIndex, const int nCols)
// {
//     bool check1 = (startIndex>=0) && (startIndex<matToModify.cols());
//     bool check2 = (startIndex+nCols)<=matToModify.cols();
//     if (check1 && check2)
//     {
//         int newColDim = matToModify.cols()-nCols;
//         Eigen::MatrixXd tmpMat = Eigen::MatrixXd::Zero(matToModify.rows(), newColDim);
//         tmpMat.leftCols(startIndex) = matToModify.leftCols(startIndex);
//         int rightHandCols = matToModify.cols() - (startIndex+nCols);
//         tmpMat.rightCols(rightHandCols) = matToModify.rightCols(rightHandCols);
//         matToModify.resize(tmpMat.rows(), tmpMat.cols());
//
//         matToModify = tmpMat;
//
//     }
//     else {
//         if (!check1) {
//             smltError("Start index (" << startIndex << ") is outside of bounds [0:"<< matToModify.cols()-1 <<"]. Please select a starting index for removal that is within these bounds.");
//         }
//         if (!check2) {
//             smltError("You are trying to remove matrix indexes ["<< startIndex<<":"<< startIndex+nCols-1 <<"] which exceed the dimension: [0:"<< matToModify.cols()-1 <<"].");
//         }
//     }
// }


const std::string currentDateTime()
{
    time_t     now = time(0);
    struct tm  tstruct;
    char       buf[80];
    tstruct = *localtime(&now);
    strftime(buf, sizeof(buf), "%Y-%m-%d.%X", &tstruct);
    return buf;
}


void checkAndCreateDirectory(const std::string dirPath)
{
    if (!boost::filesystem::exists(dirPath)) {
        boost::filesystem::create_directories(dirPath);
    }
}


void discretizeSearchSpace(Eigen::VectorXd& minVals, Eigen::VectorXd& maxVals, const int nSteps, Eigen::MatrixXd& searchSpace)
{
    Eigen::VectorXi nStepVec = Eigen::VectorXi::Constant(minVals.size(), nSteps);
    discretizeSearchSpace(minVals, maxVals, nStepVec, searchSpace);
}

void discretizeSearchSpace(Eigen::VectorXd& minVals, Eigen::VectorXd& maxVals, const Eigen::VectorXi& nStepVec, Eigen::MatrixXd& searchSpace)
{
    if ((minVals.rows()==1) && (minVals.cols()>1))
    {
        minVals.transposeInPlace();
        smltWarning("You passed a row vector. Converting to a column vector. Note that the output of this function will be in column format.");
    }
    if ((maxVals.rows()==1) && (maxVals.cols()>1))
    {
        maxVals.transposeInPlace();
        smltWarning("You passed a row vector. Converting to a column vector. Note that the output of this function will be in column format.");
    }


    int dim = minVals.rows();

    bool twoStepsMin = true;
    for (int i =0; i<dim; i++)
    {
        twoStepsMin = twoStepsMin && (nStepVec(i)>=2);
    }

    std::vector<Eigen::VectorXd> vectorsToCombine;
    vectorsToCombine.resize(dim);

    bool rowsMatch = minVals.rows()==maxVals.rows();

    if (rowsMatch && twoStepsMin)
    {
        for (int i = 0; i < dim; i++) {

            vectorsToCombine[i] = Eigen::VectorXd::LinSpaced(nStepVec(i), minVals(i), maxVals(i));
        }

        ndGrid(vectorsToCombine, searchSpace);
    }
    else {
        if (!rowsMatch) {
            smltError("Min and max vectors must be the same dimension. You passed, Min = " << minVals.rows() << "x" << minVals.cols() << " and Max = "  << maxVals.rows() << "x" << maxVals.cols() << ".");
        }
        if (!twoStepsMin) {
            smltError("Each dimension must be discretized with at least two steps in order to include the bounds. Your discretization vector was: "<< nStepVec.transpose() << ".");

        }
    }
}


void ndGrid(std::vector<Eigen::VectorXd>& vectorsToCombine, Eigen::MatrixXd& searchSpace)
{
    int nCols = vectorsToCombine.size();

    signed long long int nCombos = 1; //vectorsToCombine[0].rows();

    for(int i=0; i<nCols; i++)
    {
        nCombos *= vectorsToCombine[i].rows();
    }



    if (checkMemoryConsumption(nCols, nCombos))
    {
        searchSpace.resize(nCols, nCombos);
        Eigen::VectorXi indexVector = Eigen::VectorXi::Zero(nCols);
        int colCounter = 0;

        // This trickyness courtesy of: http://stackoverflow.com/questions/1700079/howto-create-combinations-of-several-vectors-without-hardcoding-loops-in-c#answer-1703575
        while (indexVector(0) < vectorsToCombine[0].rows())
        {
            for(int j=0; j<nCols; j++)
            {
                searchSpace(j, colCounter) = vectorsToCombine[j](indexVector(j));
            }
            indexVector(nCols-1)++;
            for (int i=nCols-1; (i>0) && (indexVector(i)==vectorsToCombine[i].rows()); i--)
            {
                indexVector(i) = 0;
                indexVector(i-1)++;
            }
            colCounter++;
        }

    }
    else
    {
        searchSpace = Eigen::MatrixXd::Zero(1,1);
    }

}

bool checkMemoryConsumption(const int nCols, const signed long long int nCombos)
{
    double memoryNeeded = nCols*nCombos* (double)sizeof(double) / 1000000000.0;
    double availibleMem = (double)sysconf(_SC_PHYS_PAGES)*(double)sysconf(_SC_PAGE_SIZE) / 1000000000.0;

    bool response = memoryNeeded<availibleMem;

    if (!response) {
        smltError("You don't have enough memory availible for all of these combinations!\n\tnumber of dimensions = " << nCols << std::endl << "\tnumber of combinations = " << nCombos << std::endl << "\tsizeof(double) = " << sizeof(double) << " bytes" <<  std::endl << "\tmemoryNeeded = " << memoryNeeded  << " Gb"<< std::endl << "\ttotal memory availible = " << availibleMem << " Gb" << std::endl);
    }

    return response;
}


};
