import numpy as np
import matplotlib.pyplot as plt
import time
mean = 1.07
stdev = 0.15
numOfYears = 25
numOfSim = 100000
initialContribution = 100
yearlyContribution = 20
inflation = 1.03

def myprint(anyval):
	pass
	#print(anyval)

yearlyReturns = stdev*np.random.randn(numOfSim,numOfYears)+mean
myprint(yearlyReturns)

#print(np.array([np.prod(yearlyReturns[:,i:],axis=1)for i in range(numOfYears)]).T)

# first element of each row = product of whole row of yearlyReturns
# second element of each row = product of [1:] whole row of yearlyReturns
# # # YearlyAccMultiplier = np.empty([numOfSim,numOfYears])
# # # t0 = time.time()
# # # for j in range(numOfSim):
	# # # for i in range(numOfYears):
		# # # YearlyAccMultiplier[j][i] = np.prod(yearlyReturns[j,i:])
		# # # #myprint(np.prod(yearlyReturns[j,i:]))
# # # myprint(YearlyAccMultiplier)
# # # t1 = time.time()
# # # total_n = t1-t0
# # # myprint(total_n)

# # #list comprehension
# # # t0 = time.time()
# # # YearlyAccMultiplier = np.array([[np.prod(yearlyReturns[j,i:]) for j in range(numOfSim)] for i in range(numOfYears)]).T
# # # #myprint(YearlyAccMultiplier)
# # # t1 = time.time()
# # # total_n = t1-t0
# # # myprint(total_n)

#list comprehension 2.0
t0 = time.time()
YearlyAccMultiplier = np.array([np.prod(yearlyReturns[:,i:],axis=1) for i in range(numOfYears)]).T
myprint(YearlyAccMultiplier)
t1 = time.time()
total_n = t1-t0
myprint(total_n)

yearlyCapitalInjection = yearlyContribution*np.ones((numOfYears,1))
yearlyCapitalInjection[0][0] = initialContribution
myprint(yearlyCapitalInjection)
inflationarr = np.array([[inflation**i for i in range(numOfYears)]]).T
myprint(inflationarr)

yearlyCapitalInjectionInflationAdjusted = inflationarr*yearlyCapitalInjection
myprint(yearlyCapitalInjectionInflationAdjusted)

distroTotalReturns = np.matmul(YearlyAccMultiplier,yearlyCapitalInjection)
myprint(distroTotalReturns)

myprint(np.mean(distroTotalReturns))
myprint(np.median(distroTotalReturns))
myprint(np.std(distroTotalReturns))
plt.hist(distroTotalReturns, normed=True, bins=100)
plt.xlabel('totalreturns')
plt.axvline(np.mean(distroTotalReturns), linestyle='--', linewidth=1.5, label="mean:"+str(np.mean(distroTotalReturns)))
plt.axvline(np.median(distroTotalReturns), linestyle='-', linewidth=1.5, label="median:"+str(np.median(distroTotalReturns)))
plt.legend(loc='upper right', shadow=True)
plt.show()