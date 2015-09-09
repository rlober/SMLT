#!/usr/bin/python

import sys
import math
import numpy as np

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D

from matplotlib import cm

rootPath = str(sys.argv[1])
print 'Looking for data directories in:\n', rootPath

path_kernelCenters = rootPath + "/gaussianProcessKernelData/kernelCenters.txt"
path_kernelTrainingData = rootPath + "/gaussianProcessKernelData/kernelTrainingData.txt"
path_maximumCovariance = rootPath + "/gaussianProcessKernelData/maximumCovariance.txt"
path_covarianceMatrix = rootPath + "/gaussianProcessKernelData/covarianceMatrix.txt"

path_inputVals = rootPath + "/gaussianProcessOutput/input.txt"
path_kernelOutput = rootPath + "/gaussianProcessOutput/kernelOutput.txt"
path_gaussianProcessMean = rootPath + "/gaussianProcessOutput/gaussianProcessMean.txt"
path_gaussianProcessVariance = rootPath + "/gaussianProcessOutput/gaussianProcessVariance.txt"
path_inputMinMax = rootPath + "/gaussianProcessOutput/inputMinMax.txt"
path_weights = rootPath + "/gaussianProcessOutput/weights.txt"



kernelCenters = np.loadtxt(path_kernelCenters)
kernelTrainingData = np.loadtxt(path_kernelTrainingData)
maximumCovariance = np.loadtxt(path_maximumCovariance)
covarianceMatrix = np.loadtxt(path_covarianceMatrix)
inputVals = np.loadtxt(path_inputVals)
kernelOutput = np.loadtxt(path_kernelOutput)
gaussianProcessMean = np.loadtxt(path_gaussianProcessMean)
gaussianProcessVariance = np.loadtxt(path_gaussianProcessVariance)
inputMinMax = np.loadtxt(path_inputMinMax)
weights = np.loadtxt(path_weights)


if len(np.shape(kernelCenters)) == 1:
    kernelDim = 1
    numberOfKernels = np.size(kernelCenters)

else:
    kernelDim, numberOfKernels = np.shape(kernelCenters)

if len(np.shape(gaussianProcessMean)) == 1:
    gpDim = 1

else:
    gpDim, numberOfSamples = np.shape(gaussianProcessMean)

print 'kernelDim: ', kernelDim
print 'numberOfKernels: ', numberOfKernels
print 'gpDim: ', gpDim
print 'weights:\n', weights
fs_labels = 16
fs_title = 20

var_alpha = 0.3
var_color = 'r'


###################################################################################
if kernelDim==1:
    fig = plt.figure(num='1D gaussian process output', figsize=(20, 8), dpi=80, facecolor='w', edgecolor='k')

    print kernelTrainingData

    for d in range(gpDim):
        ax_gp = plt.subplot2grid((3,gpDim), (0,d), rowspan=2)
        gpMean, = ax_gp.plot(inputVals[:], gaussianProcessMean[d,:], color=var_color, linewidth=4)
        gpTrain, = ax_gp.plot(kernelCenters[:], kernelTrainingData[d,:], 'bo', ms=10)

        upperVar = gaussianProcessMean[d,:] - gaussianProcessVariance[d,:]
        lowerVar = gaussianProcessMean[d,:] + gaussianProcessVariance[d,:]

        ax_gp.fill_between(inputVals, upperVar, lowerVar, alpha=var_alpha, facecolor=var_color, edgecolor=None)
        var_plot = plt.Rectangle((0, 0), 1, 1, alpha=var_alpha, facecolor=var_color, edgecolor=None)

        ax_gp.set_ylabel('G.P. output', fontsize=fs_labels)
        ax_gp.set_title('Dim: '+str(d+1))
        ax_gp.set_xlim(inputMinMax[0], inputMinMax[1])


        ax_kern = plt.subplot2grid((3,gpDim), (2,d))
        for k in range(numberOfKernels):
            tmp_kernelOutput = weights[:,d].T * kernelOutput
            kernLine, = ax_kern.plot(inputVals[:], tmp_kernelOutput[:,k], 'k--', linewidth=4)

        ax_kern.set_xlabel('Input', fontsize=fs_labels)
        ax_kern.set_ylabel('Kernel output', fontsize=fs_labels)
        ax_kern.set_xlim(inputMinMax[0], inputMinMax[1])






elif kernelDim==2:
    fig_3d = plt.figure(num='2D kernel function outputs', figsize=(20, 10), dpi=80, facecolor='w', edgecolor='k')

    nRows = int(math.sqrt(inputVals.shape[1]))
    nCols = int(inputVals.shape[1]/nRows)
    while (inputVals.shape[1] % nRows) != 0 :
        nRows += 1
        nCols = int(inputVals.shape[1]/nRows)


    print 'Reshaped matrices have nRows: ', nRows, '\tnCols: ', nCols

    Xmat = np.reshape(inputVals[0,:], (nRows, nCols))
    Ymat = np.reshape(inputVals[1,:], (nRows, nCols))

    if nRows<50:
        strideStep = 1
    elif nRows>=50 and nRows<100:
        strideStep = 5
    else:
        strideStep = 10

    for d in range(gpDim):

        ax_gp = fig_3d.add_subplot(2, gpDim, d+1, projection='3d')

        gpMeanMat = np.reshape(gaussianProcessMean[d,:], (nRows, nCols))

        surfGP = ax_gp.plot_surface(Xmat, Ymat, gpMeanMat, rstride=strideStep, cstride=strideStep, cmap=cm.cool, linewidth=1, alpha=0.6, antialiased=False)
        ax_gp.set_xlabel('Input dim 1', fontsize=fs_labels)
        ax_gp.set_ylabel('Input dim 2', fontsize=fs_labels)
        ax_gp.set_zlabel('gaussian Process mean', fontsize=fs_labels)
        ax_gp.set_xlim(inputMinMax[0,0], inputMinMax[1,0])
        ax_gp.set_ylim(inputMinMax[0,1], inputMinMax[1,1])


        fig_3d.colorbar(surfGP, shrink=0.5, aspect=5)

        ax_kern = fig_3d.add_subplot(2, gpDim, d+gpDim+1, projection='3d')


        for k in range(numberOfKernels):
            tmp_kernelOutput = weights[:,d].T * kernelOutput

            KernOutMat = np.reshape(tmp_kernelOutput[:,k], (nRows, nCols))
            surfKern = ax_kern.plot_surface(Xmat, Ymat, KernOutMat, rstride=strideStep, cstride=strideStep, cmap=cm.jet, linewidth=1, alpha=0.6, antialiased=False)

        ax_kern.set_xlabel('Input dim 1', fontsize=fs_labels)
        ax_kern.set_ylabel('Input dim 2', fontsize=fs_labels)
        ax_kern.set_zlabel('Kernel output', fontsize=fs_labels)
        ax_kern.set_xlim(inputMinMax[0,0], inputMinMax[1,0])

        fig_3d.colorbar(surfKern, shrink=0.5, aspect=5)


plt.show(block=True)
