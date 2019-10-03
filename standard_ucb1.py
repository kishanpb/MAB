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
    
    Theta = np.zeros(NO_ARMS)
    
    average_mu = 1.0*np.random.binomial(1,Mean_vector) # initial pulling of all arms
    N = np.ones(NO_ARMS)

    Regret[l-1,0:len(Mean_vector)] = Mean_vector[0]-Mean_vector

    for t in range(len(Mean_vector),T):

        reward = 0
        for i in range(len(Mean_vector)):
            Theta[i] = average_mu[i] + np.sqrt(2.0*np.log(t)/(1.0*N[i]))

        play = np.argmax(Theta)

        N[play]+=1
        average_mu[play] = average_mu[play] + \
            (np.random.binomial(1,Mean_vector[play])-average_mu[play])/(N[play])
        
        Regret[l-1,t] = Regret[l-1,t-1] + Mean_vector[0] - Mean_vector[play]

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


                