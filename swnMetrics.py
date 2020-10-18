import numpy as np
from scipy import linalg
import itertools
import matplotlib.pyplot as plt
import swnHeatKernels as swnN
import modularity as modu
import collections
import random
import os
from datetime import datetime

##########################################################################################################
############################################ BINARY NETWORKS #############################################
##########################################################################################################


# GENERATEBINARYRANDSYMADJ generates a binary random symmetric adjacency matrix
# INPUT
# vertices: number of vertices or nodes
# edges: number of edges
# OUTPUT
# A: random symmetric verticesXvertices adjacency matrix with 0s at Aij Aji if the i-j are not connected, 1 if they are

def generateBinaryRandSymAdj(vertices, edges):

    maxConnections = int(vertices * (vertices - 1) / 2)  # the maximum connections a network can have

    if edges > maxConnections or edges < 0:
        print('The number of edges are not within the permitted range')
        return -1

    # Get the indices of 1s of a matrix with 1s only on its upper triangular part
    upperTriangOnes = np.triu(np.ones((vertices, vertices)) - np.eye(vertices))
    ind = np.where(upperTriangOnes)
    # Get a random sample of those indices (#edges)
    xxRand = np.random.permutation(maxConnections)
    indRand = (ind[0][xxRand[:edges]], ind[1][xxRand[:edges]])
    # construct with those indices the upper triangular part of the adjacency matrix
    aTemp = np.zeros((vertices, vertices))
    aTemp[indRand] = 1
    # add the transpose of that matrix to get the lower part and make the matrix symmetric
    A = aTemp + np.transpose(aTemp)

    return A

# COMPCLUSTERCOEF calculates the clustering coefficient of a binary network using the formula from the Strogatz paper. Section 7.3.1 in the Networks book (2nd edition) by Newman
# Input
# A: symmetric matrix
# Output
# cCw = average clustering coefficient


def compClusterCoef(A):

    deg = np.sum(A > 0, axis=1)
    # check that there is no one or zero degree vertex
    nodesArray = np.arange(A.shape[0])
    indKeep = np.where(deg > 1)[0]
    if np.array_equal(indKeep, nodesArray) is False:
        deg = deg[indKeep]

    Nv = np.zeros(len(deg))
    for i, k in enumerate(indKeep):  # for each node that is connected more than once
        ind = np.where(A[k, :] > 0)[0]
        combs = list(itertools.combinations(ind, 2))
        # print(combs)
        counter = 0
        for ll in range(len(combs)):
            if A[combs[ll]] == 1:
                # print(combs[ll])
                counter += 1
        Nv[i] = counter

    cCAll = Nv / (0.5 * deg * (deg - 1))
    # print(cCAll)
    cC = np.sum(cCAll) / len(deg)
    return cC  # returns average clustering coefficient


# MAKEDICTFROMMATRIX takes an adjacency matrix and returns a dictionary with keys 0 to n-1 vertices, each having a list of its neighbors
# INPUT:
# A: binary adjacency matrix
# OUTPUT
# Adj: dictionary
def makeDictFromMatrix(A):

    Adj = {}
    for k in range(A.shape[0]):
        ind = np.where(A[k, :] > 0)[0]
        Adj[k] = ind

    return Adj


# BREADTHFIRSTSEARCH measures the distance of one node with the rest of the nodes in the graph using breadth first search algorithm
# INPUT:
# s: the node number for which we measure its distance with each other node
# Adj: dictionary with the connections of each node
# OUTPUT
# level: dictionary with the distances level[k] = r, r is the distance of node s to node k
def breadthFirstSearch(s, Adj):

    level = {s: 0}
    #parent = {s: 'None'}
    i = 1
    frontier = [s]  # level i-1
    while frontier:
        nextP = []  # level i
        for kk in frontier:
            for jj in Adj[kk]:  # neighbors of node kk
                if jj not in level:
                    level[jj] = i
                    # parent[jj] = kk  # parent is used to traverse the path
                    nextP.append(jj)
        frontier = nextP
        i += 1

    return level

# COMPINVLENGTHPATH computes the average inverse length path of a binary adjacency matrix using breadth first search algorithm. Computing the inverse instead of the path length is convenient for disconnected graph because then an infinite distance becomes 0 when inverting (1/infinite = 0 ). This is also called the harmonic mean distance between nodes, i.e. the average of the inverse distances 'Networks'  M. Newman, p. 172
# INPUT:
# A: adjacency matrix
# OUTPUT:
# InvPathLength: average inverse path length of each node to each other node


def compInvLengthPath(A):

    vertices = A.shape[0]
    denom = (vertices * (vertices - 1) / 2.0)
    deg = np.sum(A, axis=1)
    indZero = np.where(deg == 0)[0]
    if indZero.size > 0:
        #print('There is an adjacency matrix with no connections. We remove this row/column')
        # print(indZero)
        A = np.delete(A, indZero, axis=0)
        A = np.delete(A, indZero, axis=1)

    Adj = makeDictFromMatrix(A)
    invPathLength = 0.0
    for s in range(A.shape[0] - 1):
        level = breadthFirstSearch(s, Adj)
        for k in range(s + 1, A.shape[0]):
            if k in level:
                invPathLength += 1.0 / level[k]
                #print('inv path length between %d and %d is %f' % (s, k, 1.0 / level[k]))
            # else:
                #print('There is a disconnection between %d and %d' % (s, k))

    #print('invPathLength is %f' % invPathLength)
    avInvPathLength = invPathLength / denom

    return avInvPathLength


# COMPSMALLWORLDNESS computes the small worldness from an adjacency matrix. The formula is S = (C/Crand)*(invL/invLrand)
# INPUT:
# A:adjacency matrix
# OUTPUT:
# S: small worldness
# cc: average clustering coefficient
# invPathLength: average inverse path length. We use that so that we could compute disjoint networks. The distance between two nodes from two disjoint clusters is infinite, the inverse is 0.
def compSmallWorldness(A):

    vertices = A.shape[0]
    edges = int(np.sum(A) / 2)

    Arand = generateBinaryRandSymAdj(vertices, edges)

    ccRand = compClusterCoef(Arand)
    invPathLengthRand = compInvLengthPath(Arand)

    cc = compClusterCoef(A)
    invPathLength = compInvLengthPath(A)

    S = (cc / ccRand) * (invPathLength / invPathLengthRand)

    return (S, cc, invPathLength)


##########################################################################################################
############################################ WEIGHTED NETWORKS ###########################################
##########################################################################################################

# GENERATEWEIGHTRANDSYMADJ generates a weighted random symmetric adjacency matrix
# INPUT
# vertices: number of vertices or nodes
# edges: number of edges
# weightDistribution: it can either be normal or lognormal
# OUTPUT
# A: random symmetric verticesXvertices adjacency matrix with 0s at Aij Aji if the i-j are not connected, 0<w<1 if they are. The weights follow a distribution specified
def generateWeightRandSymAdj(vertices, edges, weightDistribution='normal'):

    maxConnections = int(vertices * (vertices - 1) / 2)  # the maximum connections a network can have
    epsilon = 0.05

    if edges > maxConnections or edges < 0:
        print('The number of edges are not within the permitted range')
        return -1

    # I use lognormal for the time being. We can make it into a parameter
    if weightDistribution == 'lognormal':
        mu, sig = 0., 1.
        randWeights = np.random.lognormal(mean=mu, sigma=sig, size=edges)
    elif weightDistribution == 'normal':
        mu, sig = 1., 0.25
        randWeights = np.random.normal(loc=mu, scale=sig, size=edges)
        ind = np.where(randWeights < 0)
        randWeights[ind] = epsilon

    # normRandWeights = randWeights / np.max(randWeights)  # normalize so that the values are between 0 and 1
    # Normalize so that the sum of the weights equals the number of edges
    normFactor = len(randWeights) / np.sum(randWeights)
    normRandWeights = randWeights * normFactor

    # Get the indices of 1s of a matrix with 1s only on its upper triangular part
    upperTriangOnes = np.triu(np.ones((vertices, vertices)) - np.eye(vertices))
    ind = np.where(upperTriangOnes)
    # Get a random sample of those indices (#edges)
    xxRand = np.random.permutation(maxConnections)
    indRand = (ind[0][xxRand[:edges]], ind[1][xxRand[:edges]])
    # construct with those indices the upper triangular part of the adjacency matrix
    aTemp = np.zeros((vertices, vertices))
    aTemp[indRand] = normRandWeights
    # add the transpose of that matrix to get the lower part and make the matrix symmetric
    A = aTemp + np.transpose(aTemp)

    return A


# COMPWEIGHTCLUSTERCOEF calculates the clustering coefficient of a weighted network using the formula from the paper. Added a 0.5 factor in the denominator to give the same result as the compClusterCoef for the binary case.
# The architecture of complex weighted networks, Barrat et al. PNAS, 2004
# Input
# A: weighted symmetric matrix
# Output
# cCw = average weighted clustering coefficient


def compWeightClusterCoef(A):

    deg = np.sum(A > 0, axis=1)
    strength = np.sum(A, axis=1)  # added weights for each node

    # check that there is no zero or one degree vertex
    nodesArray = np.arange(A.shape[0])
    indKeep = np.where(deg > 1)[0]
    if np.array_equal(indKeep, nodesArray) is False:
        print('There are %d nodes with zero or one degre' % (nodesArray.size - indKeep.size))
        deg = deg[indKeep]
        strength = strength[indKeep]

    wContrAll = np.zeros(len(deg))
    for i, k in enumerate(indKeep):  # for each node
        ind = np.where(A[k, :] > 0)[0]
        combs = list(itertools.combinations(ind, 2))
        wContr = 0.0

        for ll in range(len(combs)):
            if A[combs[ll]] > 0:
                wContr += (A[k, combs[ll][0]] + A[k, combs[ll][1]]) / 2.0

        wContrAll[i] = wContr

    cCAllw = (1 / (0.5 * strength * (deg - 1))) * wContrAll
    # print(cCAllw)
    cCw = np.sum(cCAllw) / len(deg)

    return cCw


# DIJKSTRA finds the minimum length between a source node and the rest of the nodes. The matrix is weighted with positive weights
# INPUT:
# A: adjacency matrix
# source: the source node, a number from 0 to n-1, where n is number of nodes
# OUTPUT:
# dist: an np array with the distances from the source. dist[source]=0


def dijkstra(A, source):

    vertices = A.shape[0]

    dist = np.zeros(vertices)
    dist[:] = np.inf
    dist[source] = 0
    queue = np.arange(vertices)

    leastDistNode = source
    queue = np.delete(queue, leastDistNode)
    while queue.size:  # we remove from the queue everytime the min distance node

        for neighbor in np.intersect1d(np.where(A[leastDistNode, :] > 0)[0], queue, assume_unique=True):  # Relaxation
            alt = dist[leastDistNode] + A[leastDistNode, neighbor]
            if alt < dist[neighbor]:
                dist[neighbor] = alt

        indQueue = np.argmin(dist[queue])
        leastDistNode = queue[indQueue]
        queue = np.delete(queue, indQueue)  # remove from queue

    return dist


# COMPWEIGHTINVLENGTHPATH finds the average inverse path length between all nodes in a weighted (positive weights) symmetric adjacency matrix using Dijkstra algorithm. This is also called the harmonic mean distance between nodes, i.e. the average of the inverse distances 'Networks'  M. Newman, p. 172
# INPUT:
# A: adjacency matrix
# OUTPUT:
# avPathLength: the average path length

def compWeightInvLengthPath(A):

    denom = (A.shape[0] * (A.shape[0] - 1) / 2.0)

    deg = np.sum(A > 0, axis=1)
    indZero = np.where(deg == 0)[0]
    if indZero.size > 0:
        print('There is an adjacency matrix with no connections. We remove this row/column')
        A = np.delete(A, indZero, axis=0)
        A = np.delete(A, indZero, axis=1)

    indNonZero = np.where(A > 0)
    AinvWeights = np.zeros(A.shape)
    AinvWeights[indNonZero] = 1.0 / A[indNonZero]  # we take 1 over the weights, the greater the weight the shorter the route

    vertices = A.shape[0]
    invPathLength = 0.0

    for source in range(vertices - 1):
        dist = dijkstra(AinvWeights, source)
        # print(dist[source + 1:])
        invDistances = 1.0 / dist[source + 1:]
        invPathLength += np.sum(invDistances)

    avInvPathLength = invPathLength / denom

    return avInvPathLength


# COMPWEIGHTSMALLWORLDNESS gets the small worldness from an adjacency matrix. The formula is S = (C/Crand)*(invL/invLrand)
# INPUT:
# A:adjacency matrix
# weightDistribution: the weight distribution for the random matrix
# OUTPUT:
# S: small worldness
# cc: average clustering coefficient
# inPathLength = inverse path length.We use that so that we could compute disjoint networks. The distance between two nodes from two disjoint clusters is infinite, the inverse is 0.
def compWeightSmallWorldness(A, weightDistribution):

    vertices = A.shape[0]
    edges = int(np.sum(A > 0) / 2)

    Arand = generateWeightRandSymAdj(vertices, edges, weightDistribution)
    cCRand = compWeightClusterCoef(Arand)

    invPathLengthRand = compWeightInvLengthPath(Arand)

    cC = compWeightClusterCoef(A)
    invPathLength = compWeightInvLengthPath(A)

    pathLength = 1. / invPathLength
    pathLengthRand = 1. / invPathLengthRand
    cCNorm = (cC / cCRand)
    LNorm = (pathLength / pathLengthRand)

    S = cCNorm / LNorm

    return (S, cC, invPathLength)


##########################################################################################################
############################################ ALL NETWORKS ################################################
##########################################################################################################

# REORDERA2VISUALIZE reorders the rows/columns of A so that we can see the clustering.
# It takes the eigenvector corresponding to the 2nd smallest eigenvalue of the laplacian->L = D-A and reorders it. Uses those indices to reorder the A matrix
def reorderA2Visualize(SS):

    A = SS.copy()
    deg = np.sum(A, axis=1)
    ##################################
    # if there is a degree 0, in the inversion it stays 0
    vertices = A.shape[0]
    I = np.eye(vertices)
    indDeg = np.where(deg > 0)[0]
    deg2 = np.zeros(deg.size)
    deg2[indDeg] = 1.0 / np.sqrt(deg[indDeg])

    deginv = np.expand_dims(deg2, axis=1)
    L = I - ((A * deginv) * np.transpose(deginv))  # Get the normalized Laplacian

    # decompose the matrix to its eigenvectors/eigenvalues
    eigval, eigvec = np.linalg.eigh(L)
    ##################################################

    eigSortInd = np.argsort(eigval)
    # takes second eigenvalue. The first is trivial solution. Next way to take more than one eigenvectors and do clustering
    clusterInd = np.argsort(eigvec[:, eigSortInd[1]])
    A = A[clusterInd, :]
    A = A[:, clusterInd]

    return A


# PLOTADJMATRIX reorders the rows/columns of A so that we can see the clustering and plots it
# INPUT:
# A: adjacency matrix
# cmap: the map used for plotting, Greys for binary, Hot for all other
def plotAdjMatrix(A, cmap='Greys'):

    AReord = reorderA2Visualize(A)
    plt.imshow(AReord, cmap=cmap)
    plt.show()
