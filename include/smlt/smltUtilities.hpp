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

    /*!
    *
    *   @param [in] inputMat Assumes that each row represents an sample set of random variables.
    */
    inline Eigen::MatrixXd calculateCovariance(const Eigen::MatrixXd& inputMat, bool diagonalOnly=false)
    {
        int nCols = inputMat.cols();
        Eigen::MatrixXd dispMat = inputMat - inputMat.rowwise().mean().replicate(1,nCols);

        Eigen::MatrixXd result = (dispMat * dispMat.transpose()) / (nCols-1);
        if (diagonalOnly) {
            return result.diagonal().asDiagonal();
        }else{
            return result;
        }


    };

    Eigen::VectorXd getVariance(const Eigen::MatrixXd& inputMat);

    double getVariance(const Eigen::VectorXd& inputVec);

    void removeRows(Eigen::VectorXd& vecToModify, const int startIndex, const int nRows=1);
    void removeRows(Eigen::MatrixXd& matToModify, const int startIndex, const int nRows=1);
    void removeCols(Eigen::MatrixXd& matToModify, const int startIndex, const int nCols=1);

    const std::string currentDateTime();
    void checkAndCreateDirectory(const std::string dirPath);

    Eigen::MatrixXd discretizeSearchSpace(Eigen::VectorXd& minVals, Eigen::VectorXd& maxVals, const int nSteps = 100);

    Eigen::MatrixXd ndGrid(Eigen::MatrixXd& vectorsToCombine, bool combineColWise=true);

} // End of namespace smlt

#endif
