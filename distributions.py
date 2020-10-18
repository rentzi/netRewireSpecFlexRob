import numpy as np
from scipy import linalg
import itertools
import matplotlib.pyplot as plt
import swnHeatKernels as swnN
import swnMetrics as swn
import collections
import random
import os
from datetime import datetime


# GETPERCOUTLIERDEG estimates the percent of node degrees that are outliers assuming the distribution is Poisson.
# That way we can find the heavy tailed distributions.
# INPUT:
# A: adjacency matrix
# factor: dictates the normal range aveDeg +- factor*sigma
# OUTPUT:
# percOutlier: between 0 and 1, the proportion of outlier nodes
def getPercOutlierDeg(A, factor):

    deg = np.sum(A > 0, axis=1)
    sumDeg = np.sum(np.sum(A > 0))
    aveDeg = sumDeg / A.shape[0]

    sigmaDeg = aveDeg**0.5
    spread = factor * sigmaDeg

    rangeVals = [aveDeg - spread, aveDeg + spread]

    kk = [j for j in deg if j < rangeVals[0] or j > rangeVals[1]]
    percOutlier = len(kk) / A.shape[0]

    return percOutlier


# GETPERCOUTLIERDEG estimates the percent of node degrees that are outliers assuming the distribution is Poisson.
# That way we can find the heavy tailed distributions.
# INPUT:
# A: adjacency matrix
# factor: dictates the normal range aveDeg +- factor*sigma
# OUTPUT:
# percOutlier: between 0 and 1, the proportion of outlier nodes
def getPercOutlierStrength(A, factor):

    strength = np.sum(A, axis=1)
    sumStrength = np.sum(strength)
    aveStrength = sumStrength / A.shape[0]

    sigmaStrength = aveStrength**0.5
    spread = factor * sigmaStrength

    rangeVals = [aveStrength - spread, aveStrength + spread]

    kk = [j for j in strength if j < rangeVals[0] or j > rangeVals[1]]
    percOutlier = len(kk) / A.shape[0]

    return percOutlier


# GETDERIVDATA gets the derivative from dataVec. tau is the independent variable.
# Returns derivative, length one less than dataVec

def getDerivData(dataVec, tau):

    derivData = np.zeros(len(dataVec) - 1)
    for indD, t in enumerate(np.arange(len(dataVec) - 1)):

        num = dataVec[indD + 1] - dataVec[indD]
        denom = tau[indD + 1] - tau[indD]
        derivData[indD] = num / denom

    return derivData


# GETCLUSTERCOEFTUPLES gets all the clustering metrics from the function compClusterCoefWeightAndBinarized from different combinations of taus and pRandRewires
# INPUT:
# taus: the list with the different heat diffusion parameters
# pRandRewires: the list with the probabilities of random rewiring
# rewirings: the number of iterations each time the diffusion algorithm runs
# dicMetricsIterations:  the dict with all the tuple combinations and all the iterations
# OUTPUT:
# dictClusterCoefs: a dictionary with tuple keys (pRandRewire,tau,rewirings) that stores for each tuple the a list the length of iterations with the proportions of outliers
def getPercOutliersAll(taus, pRandRewires, rewirings, dictMetricsIterations, factor=2):

    iterations = len(dictMetricsIterations)

    dictOutlierPrp = {}
    percOutliersDegAll = np.zeros((len(pRandRewires), len(taus), iterations))
    percOutliersStrengthAll = np.zeros((len(pRandRewires), len(taus), iterations))

    for indT, t in enumerate(taus):
        for indP, p in enumerate(pRandRewires):
            percOutliersDeg = np.zeros(iterations)
            percOutliersStrength = np.zeros(iterations)
            for it in np.arange(iterations):

                A = dictMetricsIterations[it + 1][(p, t, rewirings)][1]
                percOutliersDeg[it] = getPercOutlierDeg(A, factor)
                percOutliersStrength[it] = getPercOutlierStrength(A, factor)

            percOutliersDegAll[indP, indT, :] = percOutliersDeg
            percOutliersStrengthAll[indP, indT, :] = percOutliersStrength

    return percOutliersDegAll, percOutliersStrengthAll


def getDerivDataAll(taus, percOutliersDegAll, percOutliersStrengthAll):

    derivDegAll = np.zeros((percOutliersDegAll.shape[0], percOutliersDegAll.shape[1] - 1, percOutliersDegAll.shape[2]))
    derivStrengthAll = np.zeros((percOutliersDegAll.shape[0], percOutliersDegAll.shape[1] - 1, percOutliersDegAll.shape[2]))

    for p in np.arange(percOutliersDegAll.shape[0]):
        for it in np.arange(percOutliersDegAll.shape[2]):

            derivDegAll[p, :, it] = getDerivData(percOutliersDegAll[p, :, it], taus)
            derivStrengthAll[p, :, it] = getDerivData(percOutliersStrengthAll[p, :, it], taus)

    return derivDegAll, derivStrengthAll


def getDerivDataAllSingle(taus, dataAll):

    derivAll = np.zeros((dataAll.shape[0], dataAll.shape[1] - 1, dataAll.shape[2]))

    numProbs = dataAll.shape[0]
    iterations = dataAll.shape[2]
    for p in np.arange(numProbs):
        for it in np.arange(iterations):

            derivAll[p, :, it] = getDerivData(dataAll[p, :, it], taus)

    return derivAll
