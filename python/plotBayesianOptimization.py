#!/usr/bin/python
import os
import glob
import sys
import math
import numpy as np

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D

from matplotlib import cm

rootPath = str(sys.argv[1])

path_file = rootPath + '/latestLogPath.txt'

with open (path_file, "r") as myfile:
    optLogDir=myfile.readline()
# Find current optLog dir.
print 'Looking for data directories in:\n', optLogDir

searchSpace = []
searchSpaceBounds = []

gpParams = []
gpWeights = []
gpCosts = []
gpMean = []
gpVariance = []
LCB = []

minIndex = []
optimumFound = []
currentMinCost = []
currentConfidence = []
optimalParameters = []
tauFunction = []


onlyDir = [ d for d in os.listdir(optLogDir) if os.path.isdir(optLogDir + d)]
onlyDir.sort()

for iterDir in onlyDir:
    print 'Extracting data from: ', optLogDir + iterDir

    searchSpace_path        = optLogDir + iterDir + '/searchSpace.txt'
    searchSpaceBounds_path  = optLogDir + iterDir + '/searchSpaceBounds.txt'
    gpParams_path           = optLogDir + iterDir + '/gpParams.txt'
    gpWeights_path          = optLogDir + iterDir + '/gpWeights.txt'
    gpCosts_path            = optLogDir + iterDir + '/gpCosts.txt'
    gpMean_path             = optLogDir + iterDir + '/currentCostMeans.txt'
    gpVariance_path         = optLogDir + iterDir + '/currentCostVariances.txt'
    LCB_path                = optLogDir + iterDir + '/LCB.txt'
    minIndex_path           = optLogDir + iterDir + '/minIndex.txt'
    optimumFound_path       = optLogDir + iterDir + '/optimumFound.txt'
    currentMinCost_path     = optLogDir + iterDir + '/currentMinCost.txt'
    currentConfidence_path  = optLogDir + iterDir + '/currentConfidence.txt'
    optimalParameters_path  = optLogDir + iterDir + '/optimalParameters.txt'
    tau_path                = optLogDir + iterDir + '/tau.txt'

    searchSpace.append(np.loadtxt(searchSpace_path))
    searchSpaceBounds.append(np.loadtxt(searchSpaceBounds_path))
    gpParams.append(np.loadtxt(gpParams_path))
    gpWeights.append(np.loadtxt(gpWeights_path))
    gpCosts.append(np.loadtxt(gpCosts_path))
    gpMean.append(np.loadtxt(gpMean_path))
    gpVariance.append(np.loadtxt(gpVariance_path))
    LCB.append(np.loadtxt(LCB_path))
    minIndex.append(np.loadtxt(minIndex_path))
    optimumFound.append(np.loadtxt(optimumFound_path))
    currentMinCost.append(np.loadtxt(currentMinCost_path))
    currentConfidence.append(np.loadtxt(currentConfidence_path))
    optimalParameters.append(np.loadtxt(optimalParameters_path))
    tauFunction.append(np.loadtxt(tau_path))


tau = []




if len(np.shape(gpParams[0])) == 1:
    kernelDim = 1
    numberOfKernels = np.size(gpParams[0])

else:
    kernelDim, numberOfKernels = np.shape(gpParams[0])

print 'kernelDim: ', kernelDim
print 'numberOfKernels: ', numberOfKernels


fs_labels = 16
fs_title = 20

var_alpha = 0.3
var_color = 'r'

fig = plt.figure(num='1D gaussian process output', figsize=(20, 10), dpi=80, facecolor='w', edgecolor='k')

tau = []
for tmp in tauFunction:
    tau.append(tmp[()]) #weird 0D np array bug: http://stackoverflow.com/questions/773030/why-are-0d-arrays-in-numpy-not-considered-scalar

tau = np.array(tau)

tauMin = np.min(tau)*0.3
tauMax = np.max(tau)*1.3

print 'tauMin = ', tauMin, 'tauMax = ', tauMax

lcbMins = []
lcbMaxs = []

for l in LCB:
    lcbMins.append(np.min(l))
    lcbMaxs.append(np.max(l))

lcbMin = np.min(lcbMins) - np.max(lcbMaxs)*0.2
lcbMax = np.max(lcbMaxs) + np.max(lcbMaxs)*0.2
###################################################################################
if kernelDim==1:

    trueCost = np.sin(searchSpace[0]) * 3.333 +np.power(searchSpace[0], 3) /100.0 + np.power(searchSpace[0],2)/10.0
    print trueCost


    ax_gp = plt.subplot2grid((3,2), (0,0), rowspan=2)
    gpMean_line, = ax_gp.plot([], [], color=var_color, linewidth=4)
    gpTrain_line, = ax_gp.plot([], [], 'bo', ms=10)
    gpNext_line, = ax_gp.plot([], [], color='yellow', marker='v', ms=10)
    true_cost_line, = ax_gp.plot(searchSpace[0], trueCost, 'g--', linewidth=2)
    # fill_poly = ax_gp.fill_between([], [], [], alpha=var_alpha, facecolor=var_color, edgecolor=None)


    var_plot = plt.Rectangle((0, 0), 1, 1, alpha=var_alpha, facecolor=var_color, edgecolor=None)


    ax_gp.set_ylabel('Cost', fontsize=fs_labels)
    ax_gp.set_title('Gaussian Process Cost Estimate')
    ax_gp.set_xlim(searchSpaceBounds[0][0], searchSpaceBounds[0][1])


    ax_acq = plt.subplot2grid((3,2), (2,0))
    lcb_line, = ax_acq.plot([], [], linewidth=4)
    lcbMin_line, = ax_acq.plot([], [], color='yellow', marker='v', ms=10)


    ax_acq.set_ylabel('LCB', fontsize=fs_labels)
    ax_acq.set_xlabel('Input', fontsize=fs_labels)
    ax_acq.set_title(r'$\mu - \sqrt{\tau}\sigma$')
    ax_acq.set_xlim(searchSpaceBounds[0][0], searchSpaceBounds[0][1])
    ax_acq.set_ylim( lcbMin, lcbMax)

    ax_tau = plt.subplot2grid((3,2), (2,1))
    tau_line, = ax_tau.plot([], [], 'k', marker='s', ms=10, linewidth=3)

    ax_tau.set_ylabel(r'$\tau$', fontsize=fs_labels)
    ax_tau.set_xlabel('iteration', fontsize=fs_labels)
    ax_tau.set_title('tau function')
    ax_tau.set_ylim(tauMin, tauMax)
    ax_tau.set_xlim(1, len(onlyDir))


    def animate(i):
        currentMinIndex = int(minIndex[i])

        if i>0:

            gpMean_line.set_data(searchSpace[i], gpMean[i])
            gpTrain_line.set_data(gpParams[i], gpCosts[i])
            gpNext_line.set_data(searchSpace[i][currentMinIndex], gpMean[i][currentMinIndex])


            lcb_line.set_data( searchSpace[i], LCB[i])
            lcbMin_line.set_data(searchSpace[i][currentMinIndex], LCB[i][currentMinIndex])

            upperVar = gpMean[i] - gpVariance[i]
            lowerVar = gpMean[i] + gpVariance[i]

            fill_poly = ax_gp.fill_between(searchSpace[i], upperVar, lowerVar, alpha=var_alpha, facecolor=var_color, edgecolor=None)


            # if i>0:
            #
            #     upperVar = gpMean[i] - gpVariance[i]
            #     lowerVar = gpMean[i] + gpVariance[i]
            #
            #     fill_poly = ax_gp.fill_between(searchSpace[i], upperVar, lowerVar, alpha=var_alpha, facecolor=var_color, edgecolor=None)
            # else:
            #     fill_poly = ax_gp.fill_between([], [], [], alpha=var_alpha, facecolor=var_color, edgecolor=None)


            tau_line.set_data( range(1, i+2), tau[:i+1])

        else:
            gpMean_line.set_data(searchSpace[i], gpMean[i])
            gpTrain_line.set_data(gpParams[i], gpCosts[i])
            gpNext_line.set_data(searchSpace[i][currentMinIndex], gpMean[i][currentMinIndex])


            lcb_line.set_data( searchSpace[i], LCB[i])
            lcbMin_line.set_data(searchSpace[i][currentMinIndex], LCB[i][currentMinIndex])

            #
            # gpMean_line.set_data([],[])
            #
            # gpTrain_line.set_data([],[])
            #
            # gpNext_line.set_data([],[])
            #
            # lcb_line.set_data([],[])
            #
            # lcbMin_line.set_data([],[])
            fill_poly = ax_gp.fill_between([], [], [], alpha=var_alpha, facecolor=var_color, edgecolor=None)



        return gpMean_line, gpTrain_line, gpNext_line, lcb_line, lcbMin_line, tau_line, fill_poly










elif kernelDim==2:
    pass
    # fig_3d = plt.figure(num='2D kernel function outputs', figsize=(20, 10), dpi=80, facecolor='w', edgecolor='k')
    #
    # nRows = int(math.sqrt(inputVals.shape[1]))
    # nCols = int(inputVals.shape[1]/nRows)
    # while (inputVals.shape[1] % nRows) != 0 :
    #     nRows += 1
    #     nCols = int(inputVals.shape[1]/nRows)
    #
    #
    # print 'Reshaped matrices have nRows: ', nRows, '\tnCols: ', nCols
    #
    # Xmat = np.reshape(inputVals[0,:], (nRows, nCols))
    # Ymat = np.reshape(inputVals[1,:], (nRows, nCols))
    #
    # if nRows<50:
    #     strideStep = 1
    # elif nRows>=50 and nRows<100:
    #     strideStep = 5
    # else:
    #     strideStep = 10
    #
    # for d in range(gpDim):
    #
    #     ax_gp = fig_3d.add_subplot(2, gpDim, d+1, projection='3d')
    #
    #     gpMeanMat = np.reshape(gaussianProcessMean[d,:], (nRows, nCols))
    #
    #     surfGP = ax_gp.plot_surface(Xmat, Ymat, gpMeanMat, rstride=strideStep, cstride=strideStep, cmap=cm.cool, linewidth=1, alpha=0.6, antialiased=False)
    #     ax_gp.set_xlabel('Input dim 1', fontsize=fs_labels)
    #     ax_gp.set_ylabel('Input dim 2', fontsize=fs_labels)
    #     ax_gp.set_zlabel('gaussian Process mean', fontsize=fs_labels)
    #     ax_gp.set_xlim(inputMinMax[0,0], inputMinMax[1,0])
    #     ax_gp.set_ylim(inputMinMax[0,1], inputMinMax[1,1])
    #
    #
    #     fig_3d.colorbar(surfGP, shrink=0.5, aspect=5)
    #
    #     ax_kern = fig_3d.add_subplot(2, gpDim, d+gpDim+1, projection='3d')
    #
    #
    #     for k in range(numberOfKernels):
    #         tmp_kernelOutput = weights[:,d].T * kernelOutput
    #
    #         KernOutMat = np.reshape(tmp_kernelOutput[:,k], (nRows, nCols))
    #         surfKern = ax_kern.plot_surface(Xmat, Ymat, KernOutMat, rstride=strideStep, cstride=strideStep, cmap=cm.jet, linewidth=1, alpha=0.6, antialiased=False)
    #
    #     ax_kern.set_xlabel('Input dim 1', fontsize=fs_labels)
    #     ax_kern.set_ylabel('Input dim 2', fontsize=fs_labels)
    #     ax_kern.set_zlabel('Kernel output', fontsize=fs_labels)
    #     ax_kern.set_xlim(inputMinMax[0,0], inputMinMax[1,0])
    #
    #     fig_3d.colorbar(surfKern, shrink=0.5, aspect=5)


# anim = animation.FuncAnimation(fig, animate, init_func=init, frames=len(onlyDir), interval=2000, blit=True, repeat=True)
anim = animation.FuncAnimation(fig, animate, frames=len(onlyDir), interval=3000, blit=True, repeat=True)
plt.show() #block=True)
#
# file_name = './config_plots_video.mp4'
# frameRate = 1.0/(time[-1,0]/np.size(time))
# print frameRate
# anim.save(file_name, fps=frameRate, extra_args=['-vcodec', 'libx264'])
