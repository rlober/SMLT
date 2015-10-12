#ifndef SMLTUTILITIES_HPP
#define SMLTUTILITIES_HPP

#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <sstream>
#include <time.h>

#include "Eigen/Dense"

#include <boost/filesystem.hpp>

#include <unistd.h>
#include <math.h>

// #define M_PI 3.14159265359

#define smltError(errorMsg) std::cout<<"\n+++++++++++++++\n+++ [ERROR] +++\n+++++++++++++++\nfile: " << __FILE__ <<"\nfunction: " <<__func__<<"\nline: "<< __LINE__ <<"\nmessage: "<< errorMsg << "\n\n"

#define smltWarning(warningMsg) std::cout<<"\n+++++++++++++++++\n+++ [WARNING] +++\n+++++++++++++++++\nfile: " << __FILE__ <<"\nfunction: " <<__func__<<"\nline: "<< __LINE__ <<"\nmessage: "<< warningMsg << "\n\n"

namespace smlt
{

    inline Eigen::MatrixXd blockDiag(const Eigen::MatrixXd& inputMat, const int nDiagRepeats)
    {
        int nRows = inputMat.rows();
        int nCols = inputMat.cols();

        Eigen::MatrixXd output = Eigen::MatrixXd::Zero(nRows*nDiagRepeats, nCols*nDiagRepeats);
        for(int d=0; d<nDiagRepeats; d++){
            int i = nRows * d;
            int j = nCols * d;
            output.block(i,j,nRows,nCols) = inputMat;
        }


        return output;
    };



    Eigen::VectorXd getVariance(const Eigen::MatrixXd& inputMat);

    double getVariance(const Eigen::VectorXd& inputVec);

    void removeRows(Eigen::VectorXd& vecToModify, const int startIndex, const int nRows=1);
    void removeRows(Eigen::MatrixXd& matToModify, const int startIndex, const int nRows=1);
    void removeCols(Eigen::MatrixXd& matToModify, const int startIndex, const int nCols=1);

    const std::string currentDateTime();
    void checkAndCreateDirectory(const std::string dirPath);

    void discretizeSearchSpace(Eigen::VectorXd& minVals, Eigen::VectorXd& maxVals, const int nSteps, Eigen::MatrixXd& searchSpace);
    void discretizeSearchSpace(Eigen::VectorXd& minVals, Eigen::VectorXd& maxVals, const Eigen::VectorXi& nStepVec, Eigen::MatrixXd& searchSpace);


    void ndGrid(Eigen::MatrixXd& vectorsToCombine, Eigen::MatrixXd& searchSpace, bool combineColWise=true);

    /*!
    *
    *   @param [in] inputMat Assumes that each row represents an sample set of random variables.
    */
    inline Eigen::MatrixXd calculateCovariance(const Eigen::MatrixXd& inputMat, bool diagonalOnly=false)
    {
        int nCols = inputMat.cols();
        Eigen::MatrixXd result;
        if (nCols == 1) {
            Eigen::VectorXd tmpVec = inputMat.col(0);
            double bestVarianceGuess = getVariance(tmpVec);
            result = Eigen::MatrixXd::Constant(tmpVec.rows(), tmpVec.rows(), bestVarianceGuess);
        }
        else{
            Eigen::MatrixXd dispMat = inputMat - inputMat.rowwise().mean().replicate(1,nCols);
            result = (dispMat * dispMat.transpose()) / (nCols-1);

        }

        if (diagonalOnly) {
            return result.diagonal().asDiagonal();
        }else{
            return result;
        }

    };

} // End of namespace smlt

#endif
