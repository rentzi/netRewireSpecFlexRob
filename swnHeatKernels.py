import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
import swnMetrics as swn
import re


# REWIREHEATKERNEL rewires iteratively a matrix A. At each iteration the rewiring can be random (probability= pRandRewire) or according to a heat dispersion function (probability = 1-pRandRewire). Works for both binary and weighted initial networks since this implementation just redistributes the weights
# INPUT
# Arand: random symmetric adjacency matrix
# pRandRewire: probability of random rewiring
# rewirings: number of iterations the wiring take place
# tau: heat dispersion parameter
# OUTPUT
# A: returns a rewired symmetric matrix
def rewireHeatKernel(Arand, pRandRewire, rewirings, tau, eigenInd=[]):

    A = Arand.copy()

    vertices = A.shape[0]
    I = 1.0 * np.eye(vertices)

    for k in range(rewirings):

        deg = np.sum(A > 0, axis=1, keepdims=False)  # deg[i] = the number of connections of i+1 node
        vNonZeroInd = np.where((deg > 0) & (deg < vertices - 1))  # take the indices of the nodes with nonzero but not numofVertices degree
        if len(vNonZeroInd[0]) == 0:
            print('For tau = %f, and p(rand) = %f, we have graph with either fully connected or nonconnected nodes' % (tau, pRandRewire))
            return A

        vRandInd = np.random.choice(vNonZeroInd[0])  # pick one of those indices at random

        indAll = np.arange(vertices)  # 0:vertices-1
        indMinusV = np.delete(indAll, vRandInd)  # remove the vRandInd index, ie for VRandInd=2 indMinusV = 0,1,3,..
        ANotVCol = 1.0 * np.logical_not(A[indMinusV, vRandInd])  # take the actual vector and make inversions 0->1 and 1->0
        if np.random.random_sample() >= pRandRewire:  # rewire by network diffusion

            L = getNormalizedLaplacian(A)

            if len(eigenInd) == 0:
                h = linalg.expm(-tau * L)  # heat dispersion component
            else:
                h = getHeatEigenDecomp(L, tau, eigenInd)

            indTestable = np.where(A[:, vRandInd] > 0)[0]  # do not include the 0s
            # u1Testable = np.argmin(A[indTestable, vRandInd] * h[indTestable, vRandInd])  # test the nonzeros
            u1Testable = np.argmin(h[indTestable, vRandInd])  # check the heat kernel minimum value of the nodes connected to vRandInd
            u1 = indTestable[u1Testable]

            indANotVCol = np.where(ANotVCol > 0)[0]
            indNotConnected = indMinusV[indANotVCol]
            # u2IndTemp = np.argmax(ANotVCol * h[indMinusV, vRandInd])
            u2IndTemp = np.argmax(h[indNotConnected, vRandInd])  # what would happen if the ones that were connected to vRandInd were connected to it and was applied to them the heat kernel. Get the ind of the maximum from those nodes. this will be used for reconnection

            # u2 = indMinusV[u2IndTemp]  # get the right u2 node
            u2 = indNotConnected[u2IndTemp]  # get the right u2 node
        else:  # now we just randomly rewire
            noConnIndex = np.argwhere(ANotVCol)
            u2IndTemp = noConnIndex[np.random.choice(noConnIndex.size)][0]
            u2 = indMinusV[u2IndTemp]  # pick randomly a nonconnection to vRandInd node

            AOnesInd = np.argwhere(A[:, vRandInd] > 0)
            u1 = AOnesInd[np.random.choice(AOnesInd.size)][0]  # pick randomly a connected node to vRandInd

        if u1 == u2:
            print('Problem')
            print('The A nodes rewired are %d and %d with weight %f' % (u2, vRandInd, A[u1, vRandInd]))
            print('The A nodes disconnected are %d and %d' % (u1, vRandInd))
        A[u2, vRandInd] = A[u1, vRandInd]
        A[vRandInd, u2] = A[u1, vRandInd]
        A[u1, vRandInd] = 0
        A[vRandInd, u1] = 0

    return A


# rewirings is a list with rewirings. returns a dictionary Adict[rewiring[i]:A] etc
def rewireHeatKernelManyInstances(Arand, pRandRewire, rewirings, tau):

    A = Arand.copy()
    Adict = {}
    Adict[0] = Arand

    vertices = A.shape[0]
    I = 1.0 * np.eye(vertices)

    for ind, rew in enumerate(rewirings):

        if ind == 0:
            start = 0
        else:
            start = rewirings[ind - 1]

        for k in np.arange(start, rew):

            deg = np.sum(A > 0, axis=1, keepdims=False)  # deg[i] = the number of connections of i+1 node
            vNonZeroInd = np.where((deg > 0) & (deg < vertices - 1))  # take the indices of the nodes with nonzero but not numofVertices degree
            if len(vNonZeroInd[0]) == 0:
                print('For tau = %f, and p(rand) = %f, we have graph with either fully connected or nonconnected nodes' % (tau, pRandRewire))
                return A

            vRandInd = np.random.choice(vNonZeroInd[0])  # pick one of those indices at random

            indAll = np.arange(vertices)  # 0:vertices-1
            indMinusV = np.delete(indAll, vRandInd)  # remove the vRandInd index, ie for VRandInd=2 indMinusV = 0,1,3,..
            ANotVCol = 1.0 * np.logical_not(A[indMinusV, vRandInd])  # take the actual vector and make inversions 0->1 and 1->0
            if np.random.random_sample() >= pRandRewire:  # rewire by network diffusion

                L = getNormalizedLaplacian(A)

                h = linalg.expm(-tau * L)  # heat dispersion component

                indTestable = np.where(A[:, vRandInd] > 0)[0]  # do not include the 0s
                # u1Testable = np.argmin(A[indTestable, vRandInd] * h[indTestable, vRandInd])  # test the nonzeros
                u1Testable = np.argmin(h[indTestable, vRandInd])  # check the heat kernel minimum value of the nodes connected to vRandInd
                u1 = indTestable[u1Testable]

                indANotVCol = np.where(ANotVCol > 0)[0]
                indNotConnected = indMinusV[indANotVCol]
                # u2IndTemp = np.argmax(ANotVCol * h[indMinusV, vRandInd])
                u2IndTemp = np.argmax(h[indNotConnected, vRandInd])  # what would happen if the ones that were connected to vRandInd were connected to it and was applied to them the heat kernel. Get the ind of the maximum from those nodes. this will be used for reconnection

                # u2 = indMinusV[u2IndTemp]  # get the right u2 node
                u2 = indNotConnected[u2IndTemp]  # get the right u2 node
            else:  # now we just randomly rewire
                noConnIndex = np.argwhere(ANotVCol)
                u2IndTemp = noConnIndex[np.random.choice(noConnIndex.size)][0]
                u2 = indMinusV[u2IndTemp]  # pick randomly a nonconnection to vRandInd node

                AOnesInd = np.argwhere(A[:, vRandInd] > 0)
                u1 = AOnesInd[np.random.choice(AOnesInd.size)][0]  # pick randomly a connected node to vRandInd

            if u1 == u2:
                print('Problem')
                print('The A nodes rewired are %d and %d with weight %f' % (u2, vRandInd, A[u1, vRandInd]))
                print('The A nodes disconnected are %d and %d' % (u1, vRandInd))
            A[u2, vRandInd] = A[u1, vRandInd]
            A[vRandInd, u2] = A[u1, vRandInd]
            A[u1, vRandInd] = 0
            A[vRandInd, u1] = 0

        temp = A.copy()
        Adict[rew] = temp

    return Adict

# tauTuple[0] is the center of distribution, tauTuple[1] its spread, and tauTuple[2] the distribution ('normal' or 'uniform')
# except A, it also returns the taus selected from the distribution you picked


def rewireHeatKernelVariableTau(Arand, pRandRewire, rewirings, tauTuple):
    A = Arand.copy()

    vertices = A.shape[0]
    I = 1.0 * np.eye(vertices)

    tauCenter = tauTuple[0]
    tauSpread = tauTuple[1]
    tauDistribution = tauTuple[2]

    taus = generateValues(tauCenter, tauSpread, tauDistribution, rewirings)
    indNeg = np.where(taus < 0)
    taus[indNeg] = 0
    for ind, k in enumerate(np.arange(rewirings)):

        deg = np.sum(A > 0, axis=1, keepdims=False)  # deg[i] = the number of connections of i+1 node
        vNonZeroInd = np.where((deg > 0) & (deg < vertices - 1))  # take the indices of the nodes with nonzero but not numofVertices degree
        tau = taus[ind]

        if len(vNonZeroInd[0]) == 0:
            print('For tau = %f, and p(rand) = %f, we have graph with either fully connected or nonconnected nodes' % (tau, pRandRewire))
            return A

        vRandInd = np.random.choice(vNonZeroInd[0])  # pick one of those indices at random

        indAll = np.arange(vertices)  # 0:vertices-1
        indMinusV = np.delete(indAll, vRandInd)  # remove the vRandInd index, ie for VRandInd=2 indMinusV = 0,1,3,..
        ANotVCol = 1.0 * np.logical_not(A[indMinusV, vRandInd])  # take the actual vector and make inversions 0->1 and 1->0
        if np.random.random_sample() >= pRandRewire:  # rewire by network diffusion

            L = getNormalizedLaplacian(A)
            h = linalg.expm(-tau * L)  # heat dispersion component

            indTestable = np.where(A[:, vRandInd] > 0)[0]  # do not include the 0s
            # u1Testable = np.argmin(A[indTestable, vRandInd] * h[indTestable, vRandInd])  # test the nonzeros
            u1Testable = np.argmin(h[indTestable, vRandInd])  # check the heat kernel minimum value of the nodes connected to vRandInd
            u1 = indTestable[u1Testable]

            indANotVCol = np.where(ANotVCol > 0)[0]
            indNotConnected = indMinusV[indANotVCol]
            # u2IndTemp = np.argmax(ANotVCol * h[indMinusV, vRandInd])
            u2IndTemp = np.argmax(h[indNotConnected, vRandInd])  # what would happen if the ones that were connected to vRandInd were connected to it and was applied to them the heat kernel. Get the ind of the maximum from those nodes. this will be used for reconnection

            # u2 = indMinusV[u2IndTemp]  # get the right u2 node
            u2 = indNotConnected[u2IndTemp]  # get the right u2 node
        else:  # now we just randomly rewire
            noConnIndex = np.argwhere(ANotVCol)
            u2IndTemp = noConnIndex[np.random.choice(noConnIndex.size)][0]
            u2 = indMinusV[u2IndTemp]  # pick randomly a nonconnection to vRandInd node

            AOnesInd = np.argwhere(A[:, vRandInd] > 0)
            u1 = AOnesInd[np.random.choice(AOnesInd.size)][0]  # pick randomly a connected node to vRandInd

        if u1 == u2:
            print('Problem')
            print('The A nodes rewired are %d and %d with weight %f' % (u2, vRandInd, A[u1, vRandInd]))
            print('The A nodes disconnected are %d and %d' % (u1, vRandInd))
        A[u2, vRandInd] = A[u1, vRandInd]
        A[vRandInd, u2] = A[u1, vRandInd]
        A[u1, vRandInd] = 0
        A[vRandInd, u1] = 0

    return A, taus


def generateValues(center, spread, distribution, numValues):

    if distribution is 'uniform':
        values = np.random.uniform(center - spread, center + spread, numValues)
    elif distribution is 'normal':
        values = np.random.normal(center, spread, numValues)

    return values


# GETHEATEIGENDECOMP takes the L, decomposes it into its eigenvectors/values and then selects from the eigenInd the eigenvectors/values from which it will equate the heat equation h
# INPUT:
# L: the laplacian matrix
# tau: the time parameter for the heat equation
# eigenInd: the indices that indicate the eigendecomposition
# OUTPUT:
# h: heat equation


def getHeatEigenDecomp(L, tau, eigenInd):

    lambdasAll, vAll = np.linalg.eigh(L)

    lambdas = lambdasAll[eigenInd]
    v = vAll[:, eigenInd]
    vT = np.transpose(v)
    # makes a diagonal matrix from the vector
    lambdasD = np.diag(lambdas)

    hEigenval = linalg.expm(-tau * lambdasD)
    h = v@hEigenval@vT
    # print(h.shape)
    return h


def getNormalizedLaplacian(A):

    vertices = A.shape[0]
    I = np.eye(vertices)
    deg = np.sum(A, axis=1)
    # if there is a degree 0, in the inversion it stays 0
    indDeg = np.where(deg > 0)[0]
    deg2 = np.zeros(deg.size)
    deg2[indDeg] = 1.0 / np.sqrt(deg[indDeg])

    deginv = np.diag(deg2)
    L = I - deginv@A@deginv

    return L


# taus: the different taus computed and plotted ex [0.1,2,3]
# eigen: the eigenvectors plotted for example [1,3,4] the 1st, the 3rd and the 4th


def plotHeatDecompEigenTaus(L, eigen, taus):

    lambdas, v = np.linalg.eigh(L)
    vT = np.transpose(v)
    # makes a diagonal matrix from the vector
    lambdasD = np.diag(lambdas)

    cmap = 'hot'
    plt.rcParams['figure.figsize'] = [20, 20]
    totalTaus = len(taus)
    totalDecomp = len(eigen)

    for counterTau, tau in enumerate(taus):
        hEigenval = linalg.expm(-tau * lambdasD)

        hAll = v@hEigenval@vT
        plt.subplot(totalTaus, totalDecomp + 1, (totalDecomp + 1) * counterTau + 1)
        ylab = 'tau =  %.2f' % (tau)
        plt.xticks([]), plt.yticks([]), plt.ylabel(ylab)
        plt.imshow(hAll, cmap=cmap)
        plt.clim(0, 1)
        if counterTau == 0:
            plt.title('hAll')
        plt.colorbar()

        for counterEig, eig in enumerate(eigen):
            h = v[:, eig - 1:eig]@hEigenval[eig - 1:eig, eig - 1:eig]@vT[eig - 1:eig, :]
            ttl = 'h%d' % (eig)
            plt.subplot(totalTaus, totalDecomp + 1, (totalDecomp + 1) * counterTau + counterEig + 2)
            if counterTau == 0:
                plt.title(ttl)
            plt.imshow(h, cmap=cmap)
            plt.clim(0, 1)
            plt.axis('off')

    plt.show()
