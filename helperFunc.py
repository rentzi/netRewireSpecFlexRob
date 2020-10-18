import pickle
import os
import numpy as np

# SAVEVAR saves a var to a specified filePath
# INPUT:
# var: the variable to be saved

# filePath: the filepath you want to save the var, for example data/.../var.pckl. If the path does not exist, it creates it


def saveVar(var, filePath, string2Replace='reps', deleteFlag='True'):

    directory = os.path.dirname(filePath)
    if not os.path.exists(directory):  # makes the directory if it does not exist
        os.makedirs(directory)

    if os.path.isfile(filePath):  # if the file exists concatenate the data and save with the appropriate name
        var2Add = loadVar(filePath)

        if deleteFlag == 'True':
            os.remove(filePath)

        var = np.concatenate((var2Add, var), axis=2)
        string2ReplaceAll = string2Replace + str(var2Add.shape[2])
        stringReplacement = string2Replace + str(2 * var2Add.shape[2])  # because if we concatenate the data there will be 2 times the reps in them
        filePath = filePath.replace(string2ReplaceAll, stringReplacement)

    # uses pickle to serialize and save the variable var
    pickleOut = open(filePath, 'wb')
    pickle.dump(var, pickleOut)
    pickleOut.close()

# same as above, but does not concatenate if the file already exists, just replace


def saveVarSimple(var, filePath):

    directory = os.path.dirname(filePath)
    if not os.path.exists(directory):  # makes the directory if it does not exist
        os.makedirs(directory)

    # uses pickle to serialize and save the variable var
    pickleOut = open(filePath, 'wb')
    pickle.dump(var, pickleOut)
    pickleOut.close()


# LOADVAR loads a var from a specified filePath
# INPUT:
# filePath: where the variable is
# OUTPUT:
# var: the variable loaded

def loadVar(filePath):

    pickleIn = open(filePath, 'rb')
    var = pickle.load(pickleIn)
    pickleIn.close()

    return var

# CONCATDATAFROMFILESMAKEONE gets a hold of all the files with the same data, and puts the data into one file
# INPUT:
# directory: the directory where the files are
# firstPartFile: the first part of the file where it is not the identifier, ex. 'modB_reps'
# identifierStringFile: the identifier, last part of the file, ex. '_taus1_2_2_pRandRewires1_0.9_2.pckl
# the name of the whole file to be saved is firstPartFile + str(numRepetitions)+identifierStringFile
# deleteFlag: deletes all the files that were concatenated, defaults to true


def concatDataFromFilesMakeOne(directory, firstPartFile, identifierStringFile, deleteFlag='True'):

    dimAxis = 2

    targetFile = [filename for filename in os.listdir(directory) if filename.endswith(identifierStringFile) and filename.startswith(firstPartFile)]

    for idx, ff in enumerate(targetFile):

        addressLoad = directory + ff
        print(addressLoad)
        pickleIn = open(addressLoad, 'rb')
        var = pickle.load(pickleIn)
        if idx == 0:
            varAll = var
        else:
            varAll = np.concatenate((varAll, var), axis=dimAxis)

        if deleteFlag == 'True':
            os.remove(addressLoad)

        pickleIn.close()

    saveFile = directory + firstPartFile + str(varAll.shape[dimAxis]) + 'Concat' + identifierStringFile
    saveVar(varAll, saveFile)

    return varAll


# CREATEFILEPATHSTRING creates the whole path string (with the filename) that will be used to save (or load) the modularity indeces
# INPUT:
# firstPartAddress: the first part of the address without the filename
# weightDist: either 'binary' or 'normal' or 'lognormal'
# reps: the number of repetitions
# taus: a list of the taus or a single number with the tau value
# pRandRewires: a list of the pRands or a single number with the pRand value
# OUTPUT:
# filePath: the full path

def createFilePathString(firstPartAddress, weightDist, reps, taus, pRandRewires, concatFlag='True', dotEnd='.pckl', varString='mod'):

    firstPartString, identifierString = createFirstPartAndIdentifierStrings(weightDist, taus, pRandRewires, dotEnd, varString)

    if concatFlag == 'True':
        filePath = firstPartAddress + firstPartString + str(reps) + 'Concat' + identifierString
    else:
        filePath = firstPartAddress + firstPartString + str(reps) + identifierString

    return filePath


def createFirstPartAndIdentifierStrings(weightDist, taus, pRandRewires, dotEnd='.pckl', varString='mod'):

    if weightDist == 'binary':
        weightTag = 'B'
    elif weightDist == 'normal':
        weightTag = 'N'
    elif weightDist == 'lognormal':
        weightTag = 'LN'

    firstPartString = varString + weightTag + '_reps'

    if isinstance(taus, (list, np.ndarray)):
        tausString = 'taus' + str(taus[0]) + '_' + str(taus[-1]) + '_' + str(len(taus)) + '_'
    else:  # it is a number
        tausString = 'tau' + str(taus) + '_'

    if isinstance(pRandRewires, (list, np.ndarray)):
        pRandString = 'pRandRewires' + str(pRandRewires[0]) + '_' + str(pRandRewires[-1]) + '_' + str(len(pRandRewires))
    else:  # it is a number
        pRandString = 'pRandRewire' + str(pRandRewires)

    identifierString = '_' + tausString + pRandString + dotEnd

    return firstPartString, identifierString


def createFilePathStringForPlotSave(firstPartAddress, weightDist, reps, taus, pRandRewires, dotEnd='.eps', varString='mod'):

    firstPartString, identifierString = createFirstPartAndIdentifierStrings(weightDist, taus, pRandRewires, dotEnd, varString)

    filePath = firstPartAddress + firstPartString + str(reps) + identifierString

    return filePath


def createFileNameForPlotVarSlicesAllTypes(directory, reps, varNameString, listVar, dotEnd='.eps', varString='mod'):

    firstPartString = varString + 'All_reps' + str(reps) + '_'

    listString = varNameString + str(listVar[0]) + '_' + str(listVar[-1]) + '_' + str(len(listVar)) + '_'

    fileName = directory + firstPartString + listString + dotEnd

    return fileName
