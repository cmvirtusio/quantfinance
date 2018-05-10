import random
import numpy as np
import sys

def getM1(P):
    return P.mean(0)

def getM2old(P):
    meanmxP = P.mean(0)
    S = np.zeros((np.shape(P)[1], np.shape(P)[1]))
    S = np.asmatrix(S)
    for i in range(0, np.shape(P)[1]):
        for j in range(0, np.shape(P)[1]):
            u = 0;
            for t in range(0, (np.shape(P)[0] - 1)):
                u = u + ((P[t, i] - meanmxP[0, i]) * (P[t, j] - meanmxP[0, j]));
            S[i, j] = u / ((np.shape(P)[0] - 1) - 1);
    M2 = S;
    return M2

def getM3old(P):
    meanmxP = P.mean(0)
    for i in range(0, np.shape(P)[1]):
        S = np.zeros((np.shape(P)[1], np.shape(P)[1]))
        S = np.asmatrix(S)
        for j in range(0, np.shape(P)[1]):
            for k in range(0, np.shape(P)[1]):
                u = 0;
                for t in range(0, (np.shape(P)[0] - 1)):
                    u = u + ((P[t, i] - meanmxP[0, i]) * (P[t, j] - meanmxP[0, j]) * (P[t, k] - meanmxP[0, k]));
                S[j, k] = u / ((np.shape(P)[0]) - 1);
        if (i == 0):
            M3 = S
        else:
            M3 = np.concatenate((M3, S), axis=1)
    return M3

def getM4old(P):
    meanmxP = P.mean(0)
    for i in range(0, np.shape(P)[1]):
        for j in range(0, np.shape(P)[1]):
            S = np.zeros((np.shape(P)[1], np.shape(P)[1]))
            S = np.asmatrix(S)
            for k in range(0, np.shape(P)[1]):
                for l in range(0, np.shape(P)[1]):
                    u = 0;
                    for t in range(0, (np.shape(P)[0] - 1)):
                        u = u + ((P[t, i] - meanmxP[0, i]) * (P[t, j] - meanmxP[0, j]) *
                                 (P[t, k] - meanmxP[0, k]) * (P[t, l] - meanmxP[0, l]));
                    S[k, l] = u / ((np.shape(P)[0]) - 1);
            if (i == 0 and j == 0):
                M4 = S
            else:
                M4 = np.concatenate((M4, S), axis=1)
    return M4

def getM2(P):
    meanmxP = P.mean(0)
    numOfAssets = np.shape(P)[1]
    r = np.shape(P)[0]
    S = np.zeros((np.shape(P)[1], np.shape(P)[1]))
    for i in range(0, numOfAssets):
        Ti = P[:(r-1),i] - meanmxP[0, i]
        for j in range(0, numOfAssets):
            Tj = P[:(r-1),j] - meanmxP[0, j]
            S[i, j] = np.sum(np.multiply(Ti,Tj)) / ((r - 1) - 1);
    return np.asmatrix(S)
    
def getM3old2(P):
    meanmxP = P.mean(0)
    numOfAssets = np.shape(P)[1]
    r = np.shape(P)[0]
    M3 = np.zeros((numOfAssets, numOfAssets*numOfAssets))
    begCol = 0;
    endCol = numOfAssets;
    for i in range(0, numOfAssets):
        Ti = P[:(r-1), i] - meanmxP[0, i]
        S = np.zeros((numOfAssets, numOfAssets))
        for j in range(0, numOfAssets):
            Tj = P[:(r-1), j] - meanmxP[0, j]
            for k in range(0, numOfAssets):
                Tk = P[:(r-1), k] - meanmxP[0, k]
                S[j, k] = np.sum((np.multiply(np.multiply(Ti,Tj),Tk))) / (r - 1);
        M3[:,begCol:endCol] = S
        begCol=begCol+numOfAssets;
        endCol=endCol+numOfAssets;
    return np.asmatrix(M3)    
    
def getM4old2(P):
    meanmxP = P.mean(0)
    numOfAssets = np.shape(P)[1]
    r = np.shape(P)[0]
    M4 = np.zeros((numOfAssets, numOfAssets*numOfAssets*numOfAssets))
    begCol = 0;
    endCol = numOfAssets;
    for i in range(0, numOfAssets):
        Ti = P[:(r-1),i] - meanmxP[0, i]
        for j in range(0, numOfAssets):
            Tj = P[:(r-1),j] - meanmxP[0, j]
            S = np.zeros((numOfAssets, numOfAssets))
            for k in range(0, numOfAssets):
                Tk = P[:(r-1),k] - meanmxP[0, k]
                for l in range(0, numOfAssets):
                    Tl = P[:(r-1),l] - meanmxP[0, l]
                    S[k, l] = np.sum(np.multiply(np.multiply(np.multiply(Ti,Tj),Tk),Tl)) / (r - 1);
            M4[:,begCol:endCol] = S
            begCol=begCol+numOfAssets;
            endCol=endCol+numOfAssets;
    return np.asmatrix(M4)     

def getM3(P):
    meanmxP = P.mean(0)
    numOfAssets = np.shape(P)[1]
    r = np.shape(P)[0]
    M3 = np.zeros((numOfAssets, numOfAssets*numOfAssets))
    begCol = 0;
    endCol = numOfAssets;
    for i in range(0, numOfAssets):
        X = P[:(r-1),:] - meanmxP[0, :]
        S = np.zeros((numOfAssets, numOfAssets))
        S = (X.T * X)/(r-1)
        numOfData = len(X[:,i])
        diagonali = np.zeros((numOfData,numOfData))
        for rng in np.arange(numOfData):
                diagonali[rng,rng] = X[:,i].item(rng,0)
        S = (X.T * diagonali * X)/(r-1)
        M3[:,begCol:endCol] = S
        begCol=begCol+numOfAssets;
        endCol=endCol+numOfAssets;
    return np.asmatrix(M3)  
    
def getM4(P):
    meanmxP = P.mean(0)
    numOfAssets = np.shape(P)[1]
    r = np.shape(P)[0]
    M4 = np.zeros((numOfAssets, numOfAssets*numOfAssets*numOfAssets))
    begCol = 0;
    endCol = numOfAssets;
    for i in range(0, numOfAssets):
        for j in range(0, numOfAssets):
            X = P[:(r-1),:] - meanmxP[0, :]
            S = np.zeros((numOfAssets, numOfAssets))
            S = (X.T * X)/(r-1)
            numOfData = len(X[:,i])
            diagonali = np.zeros((numOfData,numOfData))
            diagonalj = np.zeros((numOfData,numOfData))
            for rng in np.arange(numOfData):
                diagonali[rng,rng] = X[:,i].item(rng,0)
                diagonalj[rng,rng] = X[:,j].item(rng,0)
            S = (X.T * diagonali * diagonalj * X)/(r-1)
            M4[:,begCol:endCol] = S
            begCol=begCol+numOfAssets;
            endCol=endCol+numOfAssets;
    return np.asmatrix(M4)    

#def fitnessfunction(Weights, M1, M2, M3, targetNotionalExposure, dailyRiskPerAsset):
def fitnessfunction(Weights, M1, M2, M3, M4, targetNotionalExposure, dailyRiskPerAsset):
    
    muP = M1 * Weights.transpose()
    sV = np.sqrt(Weights * M2 * Weights.transpose())
    sS = 0
    sK = 0
    sS = Weights * M3 * np.kron(Weights.transpose(), Weights.transpose())
    sS = -np.sign(sS) * np.power(abs(sS), (1. / 3.))
    sK = Weights * M4 * np.kron(np.kron(Weights.transpose(), Weights.transpose()), Weights.transpose())
    sK = np.power(sK, (1. / 4.))
    
    
    total = np.sum(np.absolute(Weights))

    target = targetNotionalExposure

    error = (abs(total - target) / target)
    if (error > 1):
        error = 1

    #return ((10 * dailyRiskPerAsset) + (+4 * muP - 3 * sV + 2 * sS - 1 * sK)) * (1 - error)
    
    return (1+(0.4 * muP/dailyRiskPerAsset - 0.3 * sV/dailyRiskPerAsset + 0.2 * sS/dailyRiskPerAsset - 0.1 * sK/dailyRiskPerAsset))*(1-error)

def vectorizeFunc(lsPosition, combination):
    normalizedweights = 0;
    if(lsPosition == '-1'):
        normalizedweights = (combination*0.125)-1.5
    elif(lsPosition == '1'):
        normalizedweights = (combination*0.125)+1.5
    else:
        normalizedweights = (combination*0.25)
    return normalizedweights


def hillClimbing(greatStart, iter, assets_count, corePortfolioWeights, lsPosition, notionalPerUnitArray, retmx,
                 targetNotionalExposure, dailyRiskPerAsset):
    M1 = getM1(retmx)
    M2 = getM2(retmx)
    M3 = getM3(retmx)
    M4 = getM4(retmx)
    npumx = np.asmatrix(notionalPerUnitArray)
    core = np.asmatrix(corePortfolioWeights, dtype=float)

    # look around your neighbours
    bestweights = greatStart[0]
    bestNscore = greatStart[1]
    for j in range(0, iter):
        shouldbreak = True
        bestNeighWeights = list(bestweights)
        bestNeighScore = bestNscore
        for i in range(0, assets_count):

            # add 1 to ith element
            if (bestweights[i] < 12):
                tempweights = list(bestweights)
                # print(tempweights)
                tempweights[i] = tempweights[i] + 1
                # print(tempweights)
                vfunc = np.vectorize(vectorizeFunc)
                normalizedweights = vfunc(lsPosition, tempweights)
                dWeights = np.multiply(normalizedweights, npumx)
                ww = np.asmatrix(np.column_stack((core, dWeights)))

                total = np.sum(np.absolute(ww))
                # print(total)
                target = targetNotionalExposure

                error = (abs(total - target) / target)
                # print(error)
                if (error > 0.2):
                    fitnessno = 0;
                else:
                    fitnessno = fitnessfunction(ww, M1, M2, M3, M4, targetNotionalExposure, dailyRiskPerAsset).item(0,
                                                                                                                    0)
                # print(bestNscore - fitnessno)
                if (fitnessno > bestNeighScore):
                    shouldbreak = False
                    bestNeighScore = fitnessno
                    bestNeighWeights = list(tempweights)
                    # print(bestNeighWeights)

            if (bestweights[i] > -12):
                tempweights = list(bestweights)
                # print(tempweights)
                tempweights[i] = tempweights[i] - 1
                # print(tempweights)
                vfunc = np.vectorize(vectorizeFunc)
                normalizedweights = vfunc(lsPosition, tempweights)
                dWeights = np.multiply(normalizedweights, npumx)
                ww = np.asmatrix(np.column_stack((core, dWeights)))

                total = np.sum(np.absolute(ww))
                # print(total)
                target = targetNotionalExposure

                error = (abs(total - target) / target)
                # print(error)
                if (error > 0.2):
                    fitnessno = 0;
                else:
                    fitnessno = fitnessfunction(ww, M1, M2, M3, M4, targetNotionalExposure, dailyRiskPerAsset).item(0,
                                                                                                                    0)
                # print(bestNscore - fitnessno)
                if (fitnessno > bestNscore):
                    shouldbreak = False
                    bestNeighScore = fitnessno
                    bestNeighWeights = list(tempweights)
                    # print(bestNeighWeights)
        bestweights = list(bestNeighWeights)
        bestNscore = bestNeighScore

        if (shouldbreak):
            # print(j)
            break;
    return (bestweights, bestNscore)

def genetic_algorithm(population, iterate_count, assets_count, corePortfolioWeights, lsPosition, notionalPerUnitArray, retmx, targetNotionalExposure, dailyRiskPerAsset):
    M1 = getM1(retmx)
    M2 = getM2(retmx)
    M3 = getM3(retmx)
    M4 = getM4(retmx)
    npumx = np.asmatrix(notionalPerUnitArray)
    core = np.asmatrix(corePortfolioWeights, dtype=float)

    # parents = np.array([[random.randrange(-12, 12) for _ in range(assets_count)] for _ in range(population)])

    parents = []
    while (len(parents) < population):
        # randomly select an asset
        # add a random value to that asset
        # calculate absoluteweight
        # while absolute < target
        absoluteWeight = 0;
        weightsInUnits = [];
        for directionposition in lsPosition:
            if (directionposition == '1'):
                weightsInUnits.append(-12)
            elif (directionposition == '-1'):
                weightsInUnits.append(12)
            else:
                weightsInUnits.append(0)
        fitness = 0;
        while (absoluteWeight < targetNotionalExposure):
            nthAsset = random.randint(0, assets_count - 1)
            if random.randint(0, 1) == 0:
                # deduct
                if (weightsInUnits[nthAsset] > -12):
                    weightsInUnits[nthAsset] = weightsInUnits[nthAsset] - 3
            else:
                # add
                if (weightsInUnits[nthAsset] < 12):
                    weightsInUnits[nthAsset] = weightsInUnits[nthAsset] + 3
            vfunc = np.vectorize(vectorizeFunc)
            normalizedweights = vfunc(lsPosition, weightsInUnits)
            # print(normalizedweights)
            dWeights = np.multiply(normalizedweights, npumx)
            ww = np.asmatrix(np.column_stack((core, dWeights)))
            fitness = fitnessfunction(ww, M1, M2, M3, M4, targetNotionalExposure, dailyRiskPerAsset).item(0, 0)
            absoluteWeight = np.sum(np.absolute(ww))
            # print(absoluteWeight)
        # print(weightsInUnits)
        parents.append(weightsInUnits)
        # localbest = hillClimbing((weightsInUnits,fitness),100,assets_count, corePortfolioWeights, lsPosition, notionalPerUnitArray, retmx, targetNotionalExposure, dailyRiskPerAsset)
        # parents.append(localbest[0])

    parents = np.asarray(parents)
    # print(parents)
    best = ([], -sys.maxint - 1)
    for i in range(0, iterate_count):
        fitness = []
        for x in parents:
            # normalizedweights = []
            # for y in range(0,len(x)):
            # unnormweight = x[y]
            # if(lsPosition[y] == -1):
            # normalizedweights.append((unnormweight*0.125)-1.5)
            # elif(lsPosition[y] == 1):
            # normalizedweights.append((unnormweight*0.125)+1.5)
            # else:
            # normalizedweights.append((unnormweight*0.25))
            vfunc = np.vectorize(vectorizeFunc)
            normalizedweights = vfunc(lsPosition, x)
            dWeights = np.multiply(normalizedweights, npumx)
            ww = np.asmatrix(np.column_stack((core, dWeights)))

            # total = np.sum(np.absolute(ww))

            # target = targetNotionalExposure

            # error = (abs(total - target) / target)
            # if(error > 0.2):
            # fitnessno = 0;
            # else:
            # fitnessno = fitnessfunction(ww, M1, M2, M3, M4, targetNotionalExposure, dailyRiskPerAsset).item(0, 0)
            fitnessno = fitnessfunction(ww, M1, M2, M3, M4, targetNotionalExposure, dailyRiskPerAsset).item(0, 0)
            fitness.append(fitnessno)  # fitness_function(x)
            # print(fitnessno)
        sorted_parents = []
        lex = np.argsort(fitness)
        for j in range(0, len(lex)):
            sorted_parents.append(parents[lex[j]])
        new_parents = selection(sorted_parents)
        # print(fitness[lex[-1]])
        if fitness[lex[-1]] > best[1]:
            best = (new_parents[-1], fitness[lex[-1]])
        # print best;
        children = []
        for j in range(0, population):
            r1 = random.randint(0, len(new_parents) - 1)
            r2 = random.randint(0, len(new_parents) - 1)
            children.append(crossover(parents[r1], parents[r2]))
        children = np.array(children)
        for j in range(0, len(children)):
            children[j] = mutation(children[j])
        # new_values = np.array([[random.randrange(-12, 12) * 0.25 for _ in range(assets_count)] for _ in range(n - 20)])
        # children = np.concatenate((children, new_values))
        parents = children

    best = hillClimbing(best, 100, assets_count, corePortfolioWeights, lsPosition, notionalPerUnitArray, retmx,
                        targetNotionalExposure, dailyRiskPerAsset)
    return best

def crossover(array_1, array_2):
    child = []
    for i in range(len(array_1)):
        if random.randint(0, 1) == 0:
            child.append(array_1[i])
        else:
            child.append(array_2[i])
    return np.array(child)

def mutation(array):
    for i in range(len(array)):
        if random.random() < 0.05:
            array[i] = random.randrange(-12, 12)
    return np.array(array)

def selection(array):
    toptenpercent = len(array)/5
    return array[-toptenpercent:]

