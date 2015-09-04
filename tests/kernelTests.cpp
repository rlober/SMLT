#include <iostream>
#include <ctime>
#include "Eigen/Dense"
#include "smlt/squaredExponential.hpp"

using namespace smlt;

int main(int argc, char const *argv[])
{
    std::cout << "\n\n========= BEGINING TESTS =========\n\n" << std::endl;


    Eigen::MatrixXd centers(2,2);
    centers << 1, 2,
               3, 4;

    squaredExponential kern(centers);

    kern.setCovarianceMatrix(calculateCovariance(centers, true));

    Eigen::VectorXd evalVec(2);
    evalVec << 1.5, 3.5;

    Eigen::MatrixXd output;

    std::cout << "\nChecking kernel::evaluate() for one input point:\n" << evalVec << "\n";
    kern.evaluate(evalVec, output);
    std::cout << "\noutput:\n" << output << std::endl;


    Eigen::MatrixXd evalVecs = Eigen::MatrixXd::Random(2,5);
    std::cout << "\nChecking kernel::evaluate() for multiple input points:\n" << evalVecs << "\n";
    kern.evaluate(evalVecs, output);
    std::cout << "\noutput:\n" << output << std::endl;

    std::cout << "\nGetting the designMatrix:" << std::endl;
    Eigen::MatrixXd designMatrix;
    kern.getDesignMatrix(designMatrix);
    std::cout << designMatrix << std::endl;


    std::cout << "\n\nStress tests...\n\n" << std::endl;


    Eigen::MatrixXd bigCenters  = Eigen::MatrixXd::Random(20,30);
    Eigen::MatrixXd bigSigma = calculateCovariance(bigCenters, true);
    Eigen::VectorXd bigVarVec   = Eigen::VectorXd::Ones(20);
    Eigen::MatrixXd bigEvalMat  = Eigen::MatrixXd::Random(20,200);

    Eigen::MatrixXd bigOutput;
    Eigen::MatrixXd bigDesignMat;

    kern.setCenters(bigCenters);
    // kern.setCovarianceMatrix(bigVarVec);
    kern.setCovarianceMatrix(bigSigma);

    std::cout << "Centers are " << bigCenters.rows() << "x" << bigCenters.cols() << " in dimension." << std::endl;
    std::cout << "Input is " << bigEvalMat.rows() << "x" << bigEvalMat.cols() << " in dimension." << std::endl;

    std::clock_t begin = std::clock();
    kern.evaluate(bigEvalMat, bigOutput);
    std::clock_t end = std::clock();
    double timeSec = (end - begin) / static_cast<double>( CLOCKS_PER_SEC );


    std::cout << "Output is " << bigOutput.rows() << "x" << bigOutput.cols() << " in dimension, and took " << timeSec << " seconds to evaluate." << std::endl;

    begin = std::clock();
    kern.getDesignMatrix(bigDesignMat);
    end = std::clock();
    timeSec = (end - begin) / static_cast<double>( CLOCKS_PER_SEC );

    std::cout << "Design matrix is " << bigDesignMat.rows() << "x" << bigDesignMat.cols() << " in dimension, and took " << timeSec << " seconds to evaluate." << std::endl;




    std::cout << "\n\n========= TESTS COMPLETE =========\n\n" << std::endl;


    return 0;
}
