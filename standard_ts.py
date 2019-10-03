#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

# Time horizon
T = 2*(10**3)

# Monte Carlo averaging
Mc = 10

# Number of arms
NO_ARMS = 4

# Reward vector: Size is (1 x NO_ARMS)
# First index, 0, is the best arm
Mean_vector = np.array([.5,.3,.2,.1]) 
        
Regret = np.zeros((Mc,T))

for l in range(1,Mc+1):
            
    alpha = np.ones(NO_ARMS)
    beta = np.ones(NO_ARMS)
    Theta = np.zeros(NO_ARMS)

    Mean_Estimate = alpha/(alpha+beta)
    
    for t in range(1,T+1):
        reward = 0
        for i in np.arange(1,NO_ARMS+1):
            Theta[i-1] = np.random.beta(alpha[i-1],beta[i-1])
        play = np.argmax(Theta)
        reward = np.random.binomial(1,Mean_vector[play],1)
        if (reward == 1):
            alpha[play] +=1
        else:
            beta[play] +=1
        Mean_Estimate[play] = Mean_Estimate[play]+(1/t)*(reward-Mean_Estimate[play])
                
        Regret[l-1,t-1] = Regret[l-1,t-2] + Mean_vector[0] - Mean_vector[play]

regret_avg = np.mean(Regret,axis=0)
regret_std = np.std(Regret,axis=0)
# print(regret_std.shape)

plt.figure(1)
time = np.arange(0,T,1)
plt.plot(time,(regret_avg),'--bo',markersize = 0.5)
plt.fill_between(range(len(regret_avg)),\
    regret_avg-regret_std, \
    regret_avg+regret_std, \
    color='g' )
plt.xlabel("Time")
plt.ylabel("CumulativeRegret")
# plt.legend(loc = 'upper left')
plt.show()
        
        
                