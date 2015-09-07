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

path_kernelCenters = rootPath + "/squaredExponentialKernelData/kernelCenters.txt"
path_covarianceMatrix = rootPath + "/squaredExponentialKernelData/covarianceMatrix.txt"

path_kernelInput = rootPath + "/squaredExponentialKernelOutput/kernelInput.txt"
path_kernelOutput = rootPath + "/squaredExponentialKernelOutput/kernelOutput.txt"
path_inputMinMax = rootPath + "/squaredExponentialKernelOutput/inputMinMax.txt"

kernelCenters = np.loadtxt(path_kernelCenters)
covarianceMatrix = np.loadtxt(path_covarianceMatrix)
kernelInput = np.loadtxt(path_kernelInput)
kernelOutput = np.loadtxt(path_kernelOutput)
inputMinMax = np.loadtxt(path_inputMinMax)

if len(np.shape(kernelCenters)) == 1:
    kernelDim = 1
    numberOfKernels = np.size(kernelCenters)

else:
    kernelDim, numberOfKernels = np.shape(kernelCenters)



fs_labels = 16
fs_title = 20

###################################################################################
if kernelDim==1:
    fig = plt.figure(num='1D kernel function outputs', figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')
    for k in range(numberOfKernels):
        line, = plt.plot(kernelInput[:], kernelOutput[:,k], '--', linewidth=4)

    plt.xlim(inputMinMax[0], inputMinMax[1])
    plt.xlabel('Input dim', fontsize=fs_labels)
    plt.ylabel('Kernel output', fontsize=fs_labels)





elif kernelDim==2:
    fig_3d = plt.figure(num='2D kernel function outputs', figsize=(16, 8), dpi=80, facecolor='w', edgecolor='k')
    ax_3d = fig_3d.gca(projection='3d')


    nRows = int(math.sqrt(kernelInput.shape[1]))
    nCols = int(kernelInput.shape[1]/nRows)
    while (kernelInput.shape[1] % nRows) != 0 :
        nRows += 1
        nCols = int(kernelInput.shape[1]/nRows)


    print 'Reshaped matrices have nRows: ', nRows, '\tnCols: ', nCols

    Xmat = np.reshape(kernelInput[0,:], (nRows, nCols))
    Ymat = np.reshape(kernelInput[1,:], (nRows, nCols))

    if nRows<50:
        strideStep = 1
    elif nRows>=50 and nRows<100:
        strideStep = 5
    else:
        strideStep = 10

    for k in range(numberOfKernels):
        KernOutMat = np.reshape(kernelOutput[:,k], (nRows, nCols))
        surf = ax_3d.plot_surface(Xmat, Ymat, KernOutMat, rstride=strideStep, cstride=strideStep, cmap=cm.jet, linewidth=1, alpha=0.6, antialiased=False)

    ax_3d.set_xlabel('Input dim 1', fontsize=fs_labels)
    ax_3d.set_ylabel('Input dim 2', fontsize=fs_labels)
    ax_3d.set_zlabel('Kernel output', fontsize=fs_labels)
    ax_3d.set_xlim(inputMinMax[0,0], inputMinMax[1,0])
    ax_3d.set_ylim(inputMinMax[0,1], inputMinMax[1,1])

    fig_3d.colorbar(surf, shrink=0.5, aspect=5)

if kernelDim>=2:
    fig = plt.figure(num='nD kernel function outputs projected', figsize=(16, 12), dpi=80, facecolor='w', edgecolor='k')

    if kernelDim==1:
        subPltRows = 1
        subPltCols = 1

    else:
        subPltRows = int(math.sqrt(kernelDim))
        subPltCols = int(kernelDim/subPltRows)
        while (kernelDim % subPltRows) != 0 :
            subPltRows += 1
            subPltCols = int(kernelDim/subPltRows)

    print 'subPltRows: ', subPltRows, '\tsubPltCols: ', subPltCols

    currentRow = 0
    currentCol = 0
    for d in range(kernelDim):
        ax = plt.subplot2grid((subPltRows,subPltCols), (currentRow,currentCol))
        for k in range(numberOfKernels):
            line, = ax.plot(kernelInput[d,:], kernelOutput[:,k], 'o', alpha=0.2, linewidth=0)
        ax.set_xlim(inputMinMax[0,d], inputMinMax[1,d])
        ax.set_xlabel('Input dim '+str(d), fontsize=fs_labels)
        ax.set_ylabel('Kernel output', fontsize=fs_labels)

        currentRow +=1
        if currentRow==subPltRows:
            currentRow = 0
            currentCol += 1


plt.show(block=True)
