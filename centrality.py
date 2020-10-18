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


############################################################################################################
#######################STRIPPED DOWN VERSION-WE JUST GET A AND ARAND#######################################


def getArandA(vertices, edges, tau, pRandRewire, rewirings, weightDist):

    if weightDist == 'binary':
        Arand = swn.generateBinaryRandSymAdj(vertices, edges)
    else:
        Arand = swn.generateWeightRandSymAdj(vertices, edges, weightDist)

    A = swnN.rewireHeatKernel(Arand, pRandRewire, rewirings, tau)

    return (Arand, A)


def getArandAMany(vertices, edges, taus, pRandRewires, rewirings, weightDist):

    dictMetrics = {}
    for t in taus:
        for p in pRandRewires:
            dictMetrics[(p, t, rewirings)] = getArandA(vertices, edges, t, p, rewirings, weightDist)
            print('Finished pRand = %f, tau = %f, rewirings = %d' % (p, t, rewirings))
    return dictMetrics


def getArandAManyIterations(vertices, edges, taus, pRandRewires, rewirings, weightDist, iterations):
    dictMetricsIterations = {}

    for iteration in np.arange(iterations):
        print('Iteration %d started' % (iteration + 1))
        dictMetricsIterations[iteration + 1] = getArandAMany(vertices, edges, taus, pRandRewires, rewirings, weightDist)

    return dictMetricsIterations


############################################################################################################
# WE GET ARAND AND A AT DIFFERENT ITERATIONS DICTATED BY THE VECTOR REWIRINGS
# THEY ARE STORED IN THE DICT, FOR EXAMPLE DATA[0] = ARAND, DATA[1000] = A AT 1000 ITERATIONS ETC

def getArandASteps(vertices, edges, tau, pRandRewire, rewiringVect, weightDist):

    adjMatrices = {}
    if weightDist == 'binary':
        Arand = swn.generateBinaryRandSymAdj(vertices, edges)
    else:
        Arand = swn.generateWeightRandSymAdj(vertices, edges, weightDist)

    adjMatrices[0] = Arand

    APrevious = Arand
    rwPrevious = 0
    print('start rewiring')
    for rw in rewiringVect:
        curRw = rw - rwPrevious
        adjMatrices[rw] = swnN.rewireHeatKernel(APrevious, pRandRewire, curRw, tau)

        APrevious = adjMatrices[rw]
        rwPrevious = rw
        print('Finished %d rewirings' % rwPrevious)

    return adjMatrices


######################################################################################################
######################################################################################################
############################################################################################################
#######################VARIABLE TAU WE GET A ARAND AND TAUS #######################################


def getArandAVariableTau(vertices, edges, tauTuple, pRand, rewirings, weightDist):

    if weightDist == 'binary':
        Arand = swn.generateBinaryRandSymAdj(vertices, edges)
    else:
        Arand = swn.generateWeightRandSymAdj(vertices, edges, weightDist)

    (A, taus) = swnN.rewireHeatKernelVariableTau(Arand, pRand, rewirings, tauTuple)

    return (Arand, A, taus)


def getArandAManyVariableTau(vertices, edges, tauTuples, pRand, rewirings, weightDist):

    dictMetrics = {}
    for t in tauTuples:
        dictMetrics[(pRand, t, rewirings)] = getArandAVariableTau(vertices, edges, t, pRand, rewirings, weightDist)
        print('Finished pRand = %f, tauMean = %f,tauSpread = %f,tauDistribution = %s, rewirings = %d' % (pRand, t[0], t[1], t[2], rewirings))
    return dictMetrics


def getArandAManyIterationsVariableTau(vertices, edges, tauTuples, pRand, rewirings, weightDist, iterations):
    dictMetricsIterations = {}

    for iteration in np.arange(iterations):
        print('Iteration %d started' % (iteration + 1))
        dictMetricsIterations[iteration + 1] = getArandAManyVariableTau(vertices, edges, tauTuples, pRand, rewirings, weightDist)

    return dictMetricsIterations


###########################################################################################################
def getArandA1A2(vertices, edges, tauTuple, pRand, rewiringsTuple, weightDist):

    if weightDist == 'binary':
        Arand = swn.generateBinaryRandSymAdj(vertices, edges)
    else:
        Arand = swn.generateWeightRandSymAdj(vertices, edges, weightDist)

    A1 = swnN.rewireHeatKernel(Arand, pRand, rewiringsTuple[0], tauTuple[0])
    A2 = swnN.rewireHeatKernel(A1, pRand, rewiringsTuple[1], tauTuple[1])

    return (Arand, A1, A2)


def getArandA1A2Many(vertices, edges, tauTupleList, pRand, rewiringsTuple, weightDist):

    dictMetrics = {}
    for tauTuple in tauTupleList:
        dictMetrics[(pRand, tauTuple, rewiringsTuple)] = getArandA1A2(vertices, edges, tauTuple, pRand, rewiringsTuple, weightDist)
        print('Finished pRand = %f, tau1 = %f, tau2 = %f, rewirings1 = %d, rewirings2 = %d'
              % (pRand, tauTuple[0], tauTuple[1], rewiringsTuple[0], rewiringsTuple[1]))
    return dictMetrics


def getArandA1A2ManyIterations(vertices, edges, tauTupleList, pRand, rewiringsTuple, weightDist, iterations):
    dictMetricsIterations = {}

    for iteration in np.arange(iterations):
        print('Iteration %d started' % (iteration + 1))
        dictMetricsIterations[iteration + 1] = getArandA1A2Many(vertices, edges, tauTupleList, pRand, rewiringsTuple, weightDist)

    return dictMetricsIterations
