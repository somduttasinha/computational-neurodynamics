"""
--- Computational Neurodynamics (COMP70100) ---

Solution for Exercise 2, question 3. Implements a neural controller for a
Braitenberg vehicle based on Izhikevich spiking neurons.

This script depends on the IzNetwork class, that simulates networks of
Izhikevich spiking neurons, and on the Environment class, that handles I/O from
the Braitenberg vehicle.

(C) Pedro Mediano, Murray Shanahan et al, 2016-2023
"""
import numpy as np
import numpy.random as rn

from iznetwork import IzNetwork


UMIN = 0.025            # Minimum wheel velocity in cm/ms
UMAX = UMIN + UMIN/6.0  # Maximum wheel velocity

class Controller:
    """
    Neural controller for a Braitenberg vehicle.
    """

    def __init__(self, N, dt):
        """
        Create a new robot controller comprising four populations of N neurons each.
        The controller should consist of a network of Izhikevich neurons. The robot
        takes one step every dt milliseconds of simulated Izhikevich dynamics.

        Inputs:
        N  -- Number of neurons in each population
        dt -- Number of milliseconds per robot step
        """

        self.net = self.CreateNetwork(N)
        self.dt  = dt
        self.N   = N

        self.v = np.zeros((dt, 4*N))

        # This is a reasonable value for the maximum firing rate of a single
        # excitatory neuron.
        self.Rmax = 40


    def CreateNetwork(self, N):
        """
        Construct four populations of Izhikevich neurons and connect them
        together according to the Braitenberg vehicle architecture. Each
        population has N heterogeneous regular spiking neurons.

        These should be organised so that the first N neurons are the left
        sensory population, the next N are the right sensory population, then
        the left motor, and finally the right motor population.

        For a vehicle to be attracted to light sources, sensory neurons should
        excite contralateral motor neurons causing seeking behaviour.

        Inputs:
        N -- Number of neurons in each population
        """

        F    = 50.0/np.sqrt(N)  # Scaling factor
        Dmax = 5                # Maximum conduction delay

        net = IzNetwork(4*N, Dmax)

        r = rn.rand(4*N)
        a = 0.02*np.ones(4*N)
        b = 0.2*np.ones(4*N)
        c = -65 + 15*(r**2)
        d = 8 - 6*(r**2)

        oneBlock  = np.ones((N, N))
        zeroBlock = np.zeros((N, N))

        # Block [i,j] is the connection from population i to population j
        W = np.bmat([[zeroBlock, zeroBlock,  zeroBlock, F*oneBlock],
                     [zeroBlock, zeroBlock, F*oneBlock,  zeroBlock],
                     [zeroBlock, zeroBlock,  zeroBlock,  zeroBlock],
                     [zeroBlock, zeroBlock,  zeroBlock,  zeroBlock]])

        D = Dmax*np.ones((4*N, 4*N), dtype=int)

        net.setParameters(a, b, c, d)
        net.setDelays(D)
        net.setWeights(W)

        return net


    def ControllerUpdate(self, SL, SR):
        """
        Run the internal workings of a neural controller receiving input
        sensory signals SL, SR. This function should update the neural network
        by dt ms.

        Inputs:
        SL -- Input from left sensor
        SR -- Input from right sensor

        Outputs:
        UL -- Left wheel velocity
        UR -- Right wheel velocity
        v  -- Time-by-neurons matrix of voltages during the last robot step
        """
        RL_spikes = 0.0
        RR_spikes = 0.0
        for t2 in range(self.dt):

            # Deliver stimulus as a Poisson spike stream to the sensor neurons and
            # noisy base current to the motor neurons
            I = np.hstack([rn.poisson(SL*15, self.N), rn.poisson(SR*15, self.N),
                                  3*rn.randn(self.N),        3*rn.randn(self.N)])

            # Update network
            self.net.setCurrent(I)
            fired = self.net.update()

            RL_spikes += np.sum(np.logical_and(fired > 2*self.N, fired < 3*self.N))
            RR_spikes += np.sum(fired > 3*self.N)

            # Maintain record of membrane potential
            self.v[t2,:],_ = self.net.getState()

        ## Output to motors
        # Calculate motor firing rate per neuron in Hz
        RL = 1.0*RL_spikes/(self.dt*self.N)*1000.0
        RR = 1.0*RR_spikes/(self.dt*self.N)*1000.0

        # Set wheel velocities (as fractions of UMAX)
        UL = (UMIN + (RL/self.Rmax)*(UMAX - UMIN))
        UR = (UMIN + (RR/self.Rmax)*(UMAX - UMIN))

        return UL, UR, self.v


