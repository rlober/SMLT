#include "smlt/kernelFunction.hpp"
#include "smlt/smltUtilities.hpp"

using namespace smlt;

kernelFunction::kernelFunction()
{
    // Empty constructor
}


/*!
*   Constructs the kernel and places the center coordinates according to the user input.
*   @param [in] centers The coordinates of the kernel centers. Each column should indicate the coordinates of a single center.
*/
kernelFunction::kernelFunction(const Eigen::MatrixXd& centers)
{
    centersInitialized = setCenters(centers);
    init();
}

void kernelFunction::init()
{
    doInit();
}

bool kernelFunction::setCenters(const Eigen::MatrixXd& centers)
{
    if (centers.rows()>0 && centers.cols()>0)
    {
        kernelCenters = centers;
        kernelDimension = kernelCenters.rows();
        numberOfKernels = kernelCenters.cols();
        return true;
    }
    else
    {
        // smltUtilities::smltError("Must have at least one center of at least one dimension.");
        return false;
    }
}

void kernelFunction::evaluate(const Eigen::VectorXd& inputVector, Eigen::MatrixXd& output)
{
    doEvaluate(inputVector, output);
}

void kernelFunction::evaluate(const Eigen::MatrixXd& inputVectors, Eigen::MatrixXd& output)
{
    doEvaluate(inputVectors, output);
}

void kernelFunction::getDesignMatrix(Eigen::MatrixXd& designMatrix)
{
    doEvaluate(kernelCenters, designMatrix);
}


void kernelFunction::discretizeFeasibleSet(const Eigen::VectorXd& minValues, const Eigen::VectorXd& maxValues, const int stepSize)
{
    if (minValues.rows() == maxValues.rows()) {


    }
    else{
        smltError("The min and max values do not match in dimension." << minValues.rows() << " != " << maxValues.rows() <<".");
    }
}
