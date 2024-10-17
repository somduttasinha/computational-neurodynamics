"""
--- Computational Neurodynamics (COMP70100) ---

Solutions to Exercise 2, Question 2. This script depends on the IzNetwork
class, that simulates networks of Izhikevich spiking neurons. Read through the
IzNetwork code and make sure you understand it.

Pedro Mediano, 2023
"""
import numpy as np
import numpy.random as rn
import matplotlib.pyplot as plt

from iznetwork import IzNetwork

### Question 2a
# The solution is provided in the IzNetwork class. See example uses in the
# solutions to 2b and 2c below.


### Question 2b
N = 4
Dmax = 1
net = IzNetwork(N, Dmax)

# Set connectivity and delay matrices
W = np.zeros((N,N))
D = np.ones((N,N), dtype=int)

# All neurons are heterogeneous excitatory regular spiking
r = rn.rand(N)
a = 0.02*np.ones(N)
b = 0.2*np.ones(N)
c = -65 + 15*(r**2)
d = 8 - 6*(r**2)

net.setWeights(W)
net.setDelays(D)
net.setParameters(a, b, c, d)

T = 500
V = np.zeros((T, N))
for t in range(T):
    net.setCurrent(5*np.ones(N))
    net.update()
    V[t,:], _ = net.getState()

t, n = np.where(V > 29)
plt.scatter(t, n)
plt.xlabel('Time (ms)')
plt.ylabel('Neuron index')
plt.show()


### Question 2c
# The solution is provided in the IzNetwork class. Here's a quick demonstration
# showing that the connection delays work.
N = 2
Dmax = 10
net = IzNetwork(N, Dmax)

# Set connectivity and delay matrices
W = np.array([[0, 100], [0, 0]])
D = Dmax*np.ones((N,N), dtype=int)

# All neurons are heterogeneous excitatory regular spiking
a = 0.02*np.ones(N)
b = 0.2*np.ones(N)
c = -65*np.ones(N)
d = 8*np.ones(N)

net.setWeights(W)
net.setDelays(D)
net.setParameters(a, b, c, d)

T = 60
V = np.zeros((T, N))
for t in range(T):
    net.setCurrent(np.array([10,0]))
    net.update()
    V[t,:], _ = net.getState()

plt.subplot(211)
plt.plot(V[:,0], label='Layer 1')
plt.ylabel('Voltage (mV)')
#
plt.subplot(212)
plt.plot(V[:,1], label='Layer 2')
plt.ylabel('Voltage (mV)')
plt.xlabel('Time (ms)')
plt.show()

spike_times = np.where(V > 29)[0]
assert(spike_times[1] - spike_times[0] == Dmax)

