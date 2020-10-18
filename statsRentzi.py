
import numpy as np


# MERGEFROMDICTCLUSTERCOEFDEGREES GETS ALL THE DEG CC AND CCW FOR EACH NODE FOR ALL THE ITERATIONS AND PUT THEM TOGETHER
# IT RETURNS A DICTIONARY WITH THE PARAMETERS PRANDREWIRE, TAUS, REWIRINGS WITHOUT THE ITERATIONS
def mergeFromDictClusterCoefDegrees(pRandRewires, taus, rewirings, clusterDictIterations):

    clusterDegreesDict = {}

    repetitions = len(clusterDictIterations)
    for p in pRandRewires:
        for t in taus:
            lenData = 0
            vecInd = np.zeros(repetitions)
            for rep in np.arange(repetitions):
                lenData += len(clusterDictIterations[rep + 1][(p, t, rewirings)][7])  # get the size of all the nodesDegree
                vecInd[rep] = lenData
            for rep in np.arange(repetitions):
                if rep == 0:
                    cCK = np.zeros(lenData)
                    cCKW = np.zeros(lenData)
                    deg = np.zeros(lenData)
                    startInd = 0

                cCK[int(startInd):int(vecInd[rep])] = clusterDictIterations[rep + 1][(p, t, rewirings)][7]
                cCKW[int(startInd):int(vecInd[rep])] = clusterDictIterations[rep + 1][(p, t, rewirings)][8]
                deg[int(startInd):int(vecInd[rep])] = clusterDictIterations[rep + 1][(p, t, rewirings)][9]
                startInd = vecInd[rep]

            clusterDegreesDict[(p, t, rewirings)] = (cCK, cCKW, deg, vecInd)
            # vecInd indicates when a network begins and stop, i.e. for the i-th iteration lenData[i-1]-lenData[i] are the indeces of the data for that network

    return clusterDegreesDict


# GETSDIFF gets the average differences between cCK and cCKW for different degrees  (fournd in degP) and
# for different iterations, vecInd demarcate the indices of iterations.
# it returns diff a numOfRangesXIterations matrix with the differences. You take horizontal slices for the next step
def getDiff(cCK, cCKW, deg, degP, vecInd):

    iterations = len(vecInd)

    diff = np.zeros((len(degP) + 1, iterations))
    for iteration in np.arange(iterations):
        if iteration == 0:
            startInd = 0

        endInd = vecInd[iteration]
        # gets the metrics for the iteration, data of the adjacency matrix
        cck1 = cCK[int(startInd):int(endInd)]
        cck2 = cCKW[int(startInd):int(endInd)]
        deg1 = deg[int(startInd):int(endInd)]
        #print('startInd = %d endInd=%d' % (startInd, endInd))
        startInd = endInd
        for k in np.arange(len(degP) + 1):

            if k == 0:
                indGroup = np.where(deg1 < degP[k])[0]
                #print('less than %d' % degP[k])
            elif k > 0 and k < len(degP):
                indTemp1 = np.where(deg1 >= degP[k - 1])[0]
                indTemp2 = np.where(deg1 < degP[k])[0]
                indGroup = np.intersect1d(indTemp1, indTemp2)
                #print('Between %d and %d' % (degP[k - 1], degP[k]))
            elif k == len(degP):
                indGroup = np.where(deg1 >= degP[k - 1])[0]
                #print('More than %d' % degP[k - 1])

            if indGroup.size >= 1:
                d = cck1[indGroup] - cck2[indGroup]
                if indGroup.size > 1:
                    diff[k, iteration] = np.mean(d) / (np.std(d) / np.sqrt(len(d)))
                else:
                    diff[k, iteration] = np.mean(d)
            else:
                diff[k, iteration] = np.nan
                #print('There is a range of degrees with no data')

    return diff

# GETPSIGNIFICANCE has as input an numpy vector test, that is the difference between two variables.
# The null hypothesis is that the difference is zero. It returns the p-test from a two tailed test,
# that is it checks if the value deviates either in the positive or negative way.


def getPSignificance(test, numPerms=10000, minValidIter=10):

    #nanInd = np.where(np.isnan(test))[0]
    indBoolean = np.isfinite(test)
    indNumbers = np.where(indBoolean)[0]

    #print('test is: ')
    # print(test)

    dd = test[indNumbers]
    n = len(dd)
    #print('the number of real numbers is %d' % n)

    if n >= minValidIter:
        if np.mean == 0 or np.std(dd) == 0:
            p = np.nan
            flag = ':('
            return p, flag

        observedTStat = np.mean(dd) / (np.std(dd) / np.sqrt(n))
        if numPerms > (2**n):
            print('num of permutations is not %d, it is the max possible which is %d' % (numPerms, 2**n))
            numPerms = 2**n

        x = 1 - 2 * np.random.binomial(1, .5, (numPerms, n))

        ddPerm = x * dd
        dist = np.mean(ddPerm, axis=1) / (np.std(ddPerm, axis=1) / np.sqrt(n))
        # print(observedTStat)
        if observedTStat > 0:
            p = 2 * np.mean(dist >= observedTStat)
            flag = '+'
        elif observedTStat <= 0:
            p = 2 * np.mean(dist <= observedTStat)
            flag = '-'
    else:
        p = np.nan
        flag = ':('

    return p, flag


# GETPALLFROMTUPLE gets the cc for all the iterations for one set of parameters and returns a tuple containing
# the p-values for the different groupings that we assigned it found in the degP vector
def getPAllFromTuple(cCK, cCKW, deg, vecInd, degP, numPerms=10000, minValidIter=10):

    diff = getDiff(cCK, cCKW, deg, degP, vecInd)
    numGroups = diff.shape[0]
    pAll = np.zeros(numGroups)
    flagAll = []
    for g in np.arange(numGroups):

        p, flag = getPSignificance(diff[g, :], numPerms, minValidIter)
        pAll[g] = p
        flagAll.append(flag)

    return pAll, flagAll


def getDegPSignif(pAll, degP, flagAll):

    # first get the indices that are real numbers (remove nan and inf)
    indBoolean = np.isfinite(pAll)
    indP = np.where(indBoolean)[0]
    indDeg = indP - 1  # the way the previous algorithm is constructed the corresponding deg is one index behind

    pAllValid = pAll[indP]
    flagAllValid = [flagAll[i] for i in indP]
    degAllValid = degP[indDeg]

    pSignifInd = np.where(pAllValid < 0.005)[0]
    if pSignifInd.size > 0:
        pSignif = pAllValid[pSignifInd]
        degSignif = degAllValid[pSignifInd]
        flagSignif = [flagAllValid[i] for i in pSignifInd]
    else:
        pSignif = np.empty(0)
        degSignif = np.empty(0)
        flagSignif = []

    return pSignif, flagSignif, degSignif, degAllValid


def estimateSignifPieces(degValid, degSignif, flagSignif, groupDeg, minSize=3, threshSignif=0.9):

    signifPieces = np.zeros(groupDeg.shape[0])
    for indG in np.arange(groupDeg.shape[0]):

        indDegValid1 = np.where(degValid > groupDeg[indG, 0])[0]
        indDegValid2 = np.where(degValid <= groupDeg[indG, 1])[0]
        indValid = np.intersect1d(indDegValid1, indDegValid2)

        if indValid.size > 3:
            indSignif1 = np.where(degSignif > groupDeg[indG, 0])[0]
            indSignif2 = np.where(degSignif <= groupDeg[indG, 1])[0]
            indSignif = np.intersect1d(indSignif1, indSignif2)
            if indSignif.size > 0:
                percSignif = len(indSignif) / len(indValid)
                if percSignif >= threshSignif:
                    if flagSignif[indSignif[0]] == '+':
                        signifPieces[indG] = 1
                    else:
                        signifPieces[indG] = -1
        else:
            signifPieces[indG] = 0.5

    return signifPieces


def putP2Dict(clusterDegreesDict, pRandRewires, taus, rewirings, degP, groupDeg, explanation='cc_binarized - cc_weighted'):

    clusterPStatsDict = {}
    for p in pRandRewires:
        for t in taus:
            cCK, cCKW, deg, vecInd = clusterDegreesDict[(p, t, rewirings)]

            degMin = np.min(deg)
            degMax = np.max(deg)
            degLim = [degMin, degMax]

            pAll, flagAll = getPAllFromTuple(cCK, cCKW, deg, vecInd, degP)
            pSignif, flagSignif, degSignif, degValid = getDegPSignif(pAll, degP, flagAll)

            signifPieces = estimateSignifPieces(degValid, degSignif, flagSignif, groupDeg, minSize=3, threshSignif=0.9)

            #clusterPStatsDict[(p, t, rewirings)] = (pAll, flagAll, degP, degValid, pSignif, flagSignif, degSignif, groupDeg, signifPieces, explanation)

            clusterPStatsDict[(p, t, rewirings)] = (degValid, pSignif, flagSignif, degSignif, groupDeg, signifPieces, explanation)
    return clusterPStatsDict
