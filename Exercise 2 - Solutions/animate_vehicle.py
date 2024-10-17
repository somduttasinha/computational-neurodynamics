"""
--- Computational Neurodynamics (COMP70100) ---

Auxiliary code for Exercise 2. This scripts sets up a visualisation for a
Braitenberg vehicle navigating an environment. The solution should be
provided in a separate file, controller.py as per the problem sheet.

Simulate the movement of a robot with differential wheels under the
control of a spiking neural network. The simulation runs for a very
long time --- if you get bored, press Ctrl+C a couple of times.

(C) Pedro Mediano, Murray Shanahan et al, 2016-2023
"""
import numpy as np
import numpy.random as rn
import pylab
from environment import Environment
from controller_solution import Controller

## Define simulation parameters
N    = 4      # Number of neurons in each population
NObs = 10     # Number of objects in the environment
Tmax = 20000  # Simulation time in milliseconds
dt   = 100    # Robot step size in milliseconds


## Create the environment
print('Initialising environment')
env = Environment(NObs)


## Robot controller
print('Initialising robot controller')
con = Controller(N, dt)  # <-- IMPLEMENT THIS


## Initialise record of robot positions and orientations
T = np.arange(0, Tmax, dt)
x = np.zeros(len(T)+1)
y = np.zeros(len(T)+1)
w = np.zeros(len(T)+1)
w[0] = np.pi/4


## Create Matplotlib objects for animation
print('Preparing simulation')

# Set voltage axes
fig1 = pylab.figure(1)
ax11 = fig1.add_subplot(221)
ax12 = fig1.add_subplot(222)
ax21 = fig1.add_subplot(223)
ax22 = fig1.add_subplot(224)

pl11 = ax11.plot(np.zeros((dt,N)))
ax11.set_title('Left sensory neurons')
ax11.set_ylabel('Membrane potential (mV)')
ax11.set_ylim(-90, 40)

pl12 = ax12.plot(np.zeros((dt,N)))
ax12.set_title('Right sensory neurons')
ax12.set_ylim(-90, 40)

pl21 = ax21.plot(np.zeros((dt,N)))
ax21.set_title('Left motor neurons')
ax21.set_ylabel('Membrane potential (mV)')
ax21.set_ylim(-90, 40)
ax21.set_xlabel('Time (ms)')

pl22 = ax22.plot(np.zeros((dt,N)))
ax22.set_title('Right motor neurons')
ax22.set_ylim(-90, 40)
ax22.set_xlabel('Time (ms)')

manager1 = pylab.get_current_fig_manager()

# Draw Environment
fig2 = pylab.figure(2)
ax2 = fig2.add_subplot(111)
ax2.axis([0, env.xmax, 0, env.ymax])
ax2.set_aspect(1)
ax2.set_title('Robot controlled by spiking neurons')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
for Ob in env.Obs:
    ax2.scatter(Ob['x'], Ob['y'], s=np.pi*(Ob['r']**2), c='lime')
pos_plot = ax2.scatter(x, y)

manager2 = pylab.get_current_fig_manager()

# You can change the interval duration to make the video faster or slower
timer = fig2.canvas.new_timer(interval=200)

def StopSimulation():
    global timer
    timer.stop()

t = 0

## SIMULATE
print('Start simulation')
def RobotStep(args):
    global t
  
    # Input from Sensors
    SL, SR = env.GetSensors(x[t], y[t], w[t])
  
    # Update the state of the controller and retrieve wheel velocities
    UL, UR, v = con.ControllerUpdate(SL, SR)  # <-- IMPLEMENT THIS
  
    # Update Environment
    x[t+1], y[t+1], w[t+1] = env.RobotUpdate(x[t], y[t], w[t], UL, UR, dt)
  
  
    ## PLOTTING
    for i in range(N):
        pl11[i].set_data(range(dt), v[:,i])
        pl12[i].set_data(range(dt), v[:,N+i])
  
    for i in range(N):
        pl21[i].set_data(range(dt), v[:,2*N+i])
        pl22[i].set_data(range(dt), v[:,3*N+i])
  
    # ax2.scatter(x, y)
    pos_plot.set_offsets(np.stack([x, y]).T)
    manager1.canvas.draw()
    manager2.canvas.draw()
  
    t += 1
  
    if t == len(x)-1:
        print('Terminating simulation')
        StopSimulation()

# Get the thing going
timer.add_callback(RobotStep, ())
timer.start()

pylab.show()

