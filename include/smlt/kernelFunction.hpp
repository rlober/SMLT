#ifndef KERNELFUNCTION_HPP
#define KERNELFUNCTION_HPP

#include <iostream>
#include <fstream>
#include <stdlib.h>

#include "Eigen/Dense"

namespace smlt
{
    /*! \class kernelFunction
    *   \brief A small description
    *
    *   A long description.
    *
    */

    class kernelFunction
    {
    public:
        kernelFunction();
        kernelFunction(const Eigen::MatrixXd& centers);

        bool setCenters(const Eigen::MatrixXd& centers);


        void evaluate(const Eigen::VectorXd& inputVector, Eigen::MatrixXd& output);
        void evaluate(const Eigen::MatrixXd& inputVectors, Eigen::MatrixXd& output);

        void getDesignMatrix(Eigen::MatrixXd& designMatrix);

        void discretizeFeasibleSet(const Eigen::VectorXd& minValues, const Eigen::VectorXd& maxValues, const int stepSize);

    protected:
        Eigen::MatrixXd kernelCenters; /*!< The center coordinates of the kernels. */


        int kernelDimension; /*!< The dimension of each kernel.*/
        int numberOfKernels; /*!< The total number of kernels.*/

        bool centersInitialized; /*!< A boolean to indicate whether or not the centers have been passed by the user */

        void init();

        virtual void doInit(){/*TODO: Bug here, doesn't call child implementation...*/};

        virtual void doEvaluate(const Eigen::VectorXd& inputVector, Eigen::MatrixXd& output) = 0;
        virtual void doEvaluate(const Eigen::MatrixXd& inputVectors, Eigen::MatrixXd& output) = 0;

    private:

    };
} // End of namespace smlt

#endif
