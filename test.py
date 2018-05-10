import singleAssetUtil as sAU
import multiAssetUtil as mAU
import portfolioUtil as pU

import numpy as np
import pandas_datareader as pdr
import math as math
import random
import time
from datetime import datetime
from datetime import timedelta
from pprint import pprint

print('Start')


#Initialize

#User Target Notional Exposure -- From Risk Profile
#UTNE = Number of Assets traded * average NotionalValuePerUnit, eg 10 * 3333
UTNE = 50000

#User Daily Risk Per Asset -- From Risk Profile
#User Daily Risk Per Asset - how much a user is willing to risk per day per asset (with 99% certainty)
DRPA = 100

#StartWithPortfolio -- UserInput
#StartWithPortfolio - bool if start with 0 or current portfolio 
StartWithPortfolio = False

#Current Portfolio -- UserInput
CP = np.array([[],[]])
#test for Canada
CP = np.array([['SPY','TLT','GLD','HVI.TO'],[5000,-3000,2000,-1000]])
CP = np.array([['SPY','TLT','GLD','EEM'],[5000,-3000,2000,-1000]])

if(np.shape(CP)[1] > 0):
    StartWithPortfolio = True

#Diversifier Portfolio -- UserInput
DP = np.array([[],[]])
#test for Canada
DP = np.array([['DBA','FXE','UNG','XLU','HVI.TO'],[0,-1,1,0,-1]])
#Long Only
DP = np.array([['DBA','FXE','UNG','XLU','IYR'],[1,1,1,1,1]])
#Variable
DP = np.array([['DBA','FXE','UNG','XLU','IYR'],[-1,0,1,-1,0]])
#Size 10
DP = np.array([['DBA','FXE','UNG','XLU','IYR','XLP','FXB','XIV','PCY','FXY'],[0,-1,1,0,-1,1,-1,-1,1,-1]])
#clean
DP = np.array([['DBA','FXE','UNG','XLU','IYR','XLP','FXB','XIV','PCY','FXY'],[-1,-1,-1,-1,-1,1,1,1,1,1]])


########################USER_INPUT########################




#Weights of Each Assets in Core Portfolio
corePortWeight = np.array([]);
#percentreturns Matrix of Core Assets
retmxCP = []
#percentreturns Matrix of Diversifier Assets
retmxDP = []
#NotionalValuePerUnit matrix of Diversifier Assets (precalculated)
npu = []


#If user has portfolio, initialize corePortWeight and retmxCP
if(StartWithPortfolio):
    corePortWeight = np.append(corePortWeight,CP[1,:])
    for ticker in np.array(CP[0,:]):
        s = pdr.data.get_data_yahoo(symbols=ticker, start=datetime.now().date() - timedelta(days=365), end=datetime.now().date(), interval='d')
        ret = sAU.get_period_returns(s)
        retmxCP.append(ret[::-1])
#pprint(retmxCP)
#pprint(corePortWeight)

#return npu and retmxDP
for x in range(0,len(DP[0])):
    ticker = DP[0][x]
    s = pdr.data.get_data_yahoo(symbols=ticker, start=datetime.now().date() - timedelta(days=365), end=datetime.now().date(), interval='d')
    ret = sAU.get_period_returns(s)
    npu.append(sAU.get_notional_value_per_unit(ret, s.iloc[-1]['Adj Close'], len(s), DRPA))
    retmxDP.append(ret[::-1])
#pprint(npu)
#pprint(retmxDP)

#Combine Core Portfolio, Diversifier Portfolio return matrix
############################################################
CPmx = np.asmatrix(retmxCP).transpose()
#pprint(CPmx)
DPmx = np.asmatrix(retmxDP).transpose()
#pprint(DPmx)

if(StartWithPortfolio):
    retmx = np.concatenate((CPmx,DPmx), axis =1)
else:
    retmx = DPmx
#pprint(retmx)
############################################################

#Time GA

# np.set_printoptions(threshold='nan')
# start = time.time()
# print(1000000000*pU.getM3old(retmx))
# done = time.time()
# elapsed = done - start
# print(elapsed)
# start = time.time()
# print(1000000000*pU.getM3new(retmx))
# done = time.time()
# elapsed = done - start
# print(elapsed)
# print('STOP')
# print(1000000000*pU.getM3old(retmx) - 1000000000*pU.getM3new(retmx))


for i in range(0,10):
    print((i+3)*10)
    for x in range(0,7):
        start = time.time()
        population = (i+3)*10
        iteration = 10000/population
        (combination,fitnessScore) = pU.genetic_algorithm(population, iteration, np.shape(DP)[1], corePortWeight, (DP)[1], npu, retmx, UTNE, DRPA)
        normalizedweights = []
        for y in range(0,len(combination)):
            unnormweight = combination[y]
            if((DP)[1][y] == '-1'):
                normalizedweights.append((unnormweight*0.125)-1.5)
            elif((DP)[1][y] == '1'):
                normalizedweights.append((unnormweight*0.125)+1.5)
            else:
                normalizedweights.append((unnormweight*0.25))
        pprint(normalizedweights)
        print(fitnessScore)
        done = time.time()
        elapsed = done - start
        print(elapsed)
    print('--------------------------------------------------------------')
print('End')