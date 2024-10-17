"""
--- Computational Neurodynamics (COMP70100) ---

Template for Exercise 2, question 3. Implements a neural controller for a
Braitenberg vehicle based on Izhikevich spiking neurons.

This script depends on the Environment class, that handles I/O from
the Braitenberg vehicle.

(C) Pedro Mediano, Murray Shanahan et al, 2016-2023
"""
import numpy as np
import numpy.random as rn


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

        raise NotImplementedError


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

        raise NotImplementedError

