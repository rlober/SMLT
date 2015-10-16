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


/*
These template functions aren't very robust -> need to rewrite them to reflect the eigen guidlines.
*/





template <typename T>
Eigen::MatrixXd blockDiag(const T& base, const int nRepeats)
{
    int nRows = base.rows();
    int nCols = base.cols();
    Eigen::MatrixXd tmpMat = Eigen::MatrixXd::Zero(nRows*nRepeats, nCols*nRepeats);

    for(int d=0; d<nRepeats; d++){
            int i = nRows * d;
            int j = nCols * d;
            tmpMat.block(i,j,nRows,nCols) = base;
    }

    return tmpMat;
}

template <typename T1, typename T2>
Eigen::MatrixXd hStack(T1& base, const T2& appendee)
{
    int nRows = base.rows();
    int nCols = base.cols() + appendee.cols();

    Eigen::MatrixXd concat = Eigen::MatrixXd::Zero(nRows, nCols);

    if (base.rows()==appendee.rows()) {concat << base, appendee;}
    else {
        std::cout << "The number of rows do not match: " << base.rows() << " and " << appendee.rows() << ". Returning null matrix." << std::endl;
    }

    return concat;
}

template <typename T1, typename T2>
T1 vStack(T1& base, const T2& appendee)
{
    int nRows = base.rows() + appendee.rows();
    int nCols = base.cols();

    T1 concat;
    concat.resize(nRows, nCols);


    if (base.cols()==appendee.cols()) {concat << base, appendee;}
    else {
        std::cout << "The number of columns do not match: " << base.cols() << " and " << appendee.cols() << ". Returning null matrix." << std::endl;
    }

    return concat;

}


template <typename T>
void removeRows(T& base, const int startIndex, const int nRows)
{
    bool check1 = (startIndex>=0) && (startIndex<base.rows());
    bool check2 = (startIndex+nRows)<=base.rows();
    if (check1 && check2)
    {
        int newRowDim = base.rows()-nRows;
        T tmpMat;
        tmpMat.resize(newRowDim, base.cols());

        tmpMat.topRows(startIndex) = base.topRows(startIndex);
        int bottomRows = base.rows() - (startIndex+nRows);
        tmpMat.bottomRows(bottomRows) = base.bottomRows(bottomRows);
        base.resize(tmpMat.rows(), tmpMat.cols());

        base = tmpMat;

    }
    else {
        if (!check1) {
            smltError("Start index (" << startIndex << ") is outside of bounds [0:"<< base.rows()-1 <<"]. Please select a starting index for removal that is within these bounds.");
        }
        if (!check2) {
            smltError("You are trying to remove matrix indexes ["<< startIndex<<":"<< startIndex+nRows-1 <<"] which exceed the dimension: [0:"<< base.rows()-1 <<"].");
        }
    }
}

template <typename T>
void removeCols(T& base, const int startIndex, const int nCols)
{
    bool check1 = (startIndex>=0) && (startIndex<base.cols());
    bool check2 = (startIndex+nCols)<=base.cols();
    if (check1 && check2)
    {
        int newColDim = base.cols()-nCols;
        T tmpMat;
        tmpMat.resize(base.rows(), newColDim);

        tmpMat.leftCols(startIndex) = base.leftCols(startIndex);
        int rightHandCols = base.cols() - (startIndex+nCols);
        tmpMat.rightCols(rightHandCols) = base.rightCols(rightHandCols);
        base.resize(tmpMat.rows(), tmpMat.cols());

        base = tmpMat;

    }
    else {
        if (!check1) {
            smltError("Start index (" << startIndex << ") is outside of bounds [0:"<< base.cols()-1 <<"]. Please select a starting index for removal that is within these bounds.");
        }
        if (!check2) {
            smltError("You are trying to remove matrix indexes ["<< startIndex<<":"<< startIndex+nCols-1 <<"] which exceed the dimension: [0:"<< base.cols()-1 <<"].");
        }
    }
}




template <typename T1, typename T2>
void insertRows(T1& base, const T2& rowsToInsert, const int insertIndex)
{
    int nRows = rowsToInsert.rows();

    bool check1 = (insertIndex>0) && (insertIndex<base.rows());
    bool check2 = base.cols() == rowsToInsert.cols();

    if (check1 && check2)
    {
        int newRowDim = base.rows()+nRows;
        T1 tmpMat;
        tmpMat.resize(newRowDim, base.cols());

        tmpMat.topRows(insertIndex) = base.topRows(insertIndex);
        tmpMat.middleRows(insertIndex, nRows) = rowsToInsert;

        int bottomRows = base.rows() - insertIndex;
        tmpMat.bottomRows(bottomRows) = base.bottomRows(bottomRows);
        base.resize(tmpMat.rows(), tmpMat.cols());

        base = tmpMat;

    }
    else {
        if (!check1) {
            smltError("Start index (" << insertIndex << ") is outside of bounds [0:"<< base.rows()-1 <<"]. Please select a starting index for removal that is within these bounds.");
        }
        if (!check2) {
            smltError("The column dimensions of the inserted rows ("<< rowsToInsert.cols() <<") must match that of the base vector or matrix ("<< base.cols() <<"). Doing nothing.");
        }
    }
}

template <typename T1, typename T2>
void insertCols(T1& base, const T2& colsToInsert, const int insertIndex)
{
    int nCols = colsToInsert.cols();


    bool check1 = (insertIndex>0) && (insertIndex<base.cols());
    bool check2 = base.rows() == colsToInsert.rows();


    if (check1 && check2)
    {
        int newColDim = base.cols()+nCols;
        T1 tmpMat;
        tmpMat.resize(base.rows(), newColDim);

        tmpMat.leftCols(insertIndex) = base.leftCols(insertIndex);
        tmpMat.middleCols(insertIndex, nCols) = colsToInsert;

        int rightHandCols = base.cols() - insertIndex;
        tmpMat.rightCols(rightHandCols) = base.rightCols(rightHandCols);
        base.resize(tmpMat.rows(), tmpMat.cols());

        base = tmpMat;

    }
    else {
        if (!check1) {
            smltError("Start index (" << insertIndex << ") is outside of bounds [0:"<< base.cols()-1 <<"]. Please select a starting index for removal that is within these bounds.");
        }
        if (!check2) {
            smltError("The row dimensions of the inserted columns ("<< colsToInsert.rows() <<") must match that of the base vector or matrix ("<< base.rows() <<"). Doing nothing.");
        }
    }
}



namespace smlt
{

    Eigen::VectorXd getVariance(const Eigen::MatrixXd& inputMat);

    double getVariance(const Eigen::VectorXd& inputVec);

    // void removeRows(Eigen::VectorXd& vecToModify, const int startIndex, const int nRows=1);
    // void removeRows(Eigen::MatrixXd& matToModify, const int startIndex, const int nRows=1);
    // void removeCols(Eigen::MatrixXd& matToModify, const int startIndex, const int nCols=1);

    const std::string currentDateTime();
    void checkAndCreateDirectory(const std::string dirPath);

    void discretizeSearchSpace(Eigen::VectorXd& minVals, Eigen::VectorXd& maxVals, const int nSteps, Eigen::MatrixXd& searchSpace);
    void discretizeSearchSpace(Eigen::VectorXd& minVals, Eigen::VectorXd& maxVals, const Eigen::VectorXi& nStepVec, Eigen::MatrixXd& searchSpace);


    void ndGrid(std::vector<Eigen::VectorXd>& vectorsToCombine, Eigen::MatrixXd& searchSpace);


    bool checkMemoryConsumption(const int nCols, const signed long long int nCombos);


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
