"""
--- Computational Neurodynamics (COMP70100) ---

Auxiliary code for Exercise 2. Implements functions needed for a Braitenberg
vehicle to interact with its environment.

(C) Pedro Mediano, Murray Shanahan et al, 2016-2023
"""
import numpy as np
import numpy.random as rn


class Environment:
  """
  Environment for the robot to run around. Holds a list of objects
  the robot should either avoid or catch.
  """

  def __init__(self, _Obs, _MinSize=10, _MaxSize=20, _xmax=100, _ymax=100):
    """
    Create a new environment comprising a list of length Obs of objects.  Each
    new object is assigned a random size (between MinSize and MaxSize) and a
    random location on a torus. xmax and ymax are the limits of the torus. In
    the current implementation, the size of the object does not change the
    effect on the robot's sensors.

    Inputs:
    _Obs     -- Number of objects to draw in the environment
    _MinSize -- Minimum size of objects
    _MaxSize -- Maximum size of objects
    _xmax    -- Maximum X position of objects
    _ymax    -- Maximum Y position of objects
    """

    self.xmax = _xmax
    self.ymax = _ymax

    self.Obs = [{'x': rn.rand()*_xmax,
                 'y': rn.rand()*_ymax,
                 'r': _MinSize + rn.rand()*(_MaxSize - _MinSize)
                 } for _ in range(_Obs)
                ]

  def GetSensors(self, x, y, w):
    """
    Return the current activities of the robot's sensors given
    its position (x,y) and orientation w.
    All geometry is calculated on a torus with limits xmax and ymax.

    Inputs:
    x, y, w -- Position and orientation of robot

    Outputs:
    SL, SR -- Activities of left and right sensor
    """

    SL = 0
    SR = 0
    Range = 25.0  # Sensor range

    for Ob in self.Obs:
      x2 = Ob['x']
      y2 = Ob['y']

      # Find the shortest x distance on torus
      if abs(x2 + self.xmax - x) < abs(x2 - x):
        x2 = x2 + self.xmax
      elif abs(x2 - self.xmax - x) < abs(x2 - x):
        x2 = x2 - self.xmax

      # Find shortest y distance on torus
      if abs(y2 + self.ymax - y) < abs(y2 - y):
        y2 = y2 + self.ymax
      elif abs(y2 - self.ymax - y) < abs(y2 - y):
        y2 = y2 - self.ymax

      dx = x2 - x
      dy = y2 - y

      z = np.sqrt(dx**2 + dy**2)

      if z < Range:
        v = np.arctan2(dy, dx)
        if v < 0:
          v = 2*np.pi + v

        dw = v - w  # angle difference between robot's heading and object

        # Stimulus strength depends on distance to object boundary
        S = (Range - z)/Range

        # There are two options here: either the robot only sees the
        # "brightest" object (max) or it sees a superposition of all the
        # objects (sum). The "sum" option seems to yield nicer trajectories, so
        # the other option is left commented out.
        if ((dw >= np.pi/8 and dw < np.pi/2) or
                (dw < -1.5*np.pi and dw >= -2*np.pi+np.pi/8)):
          # SL = max(S, SL)
          SL += S
        elif ((dw > 1.5*np.pi and dw <= 2*np.pi - np.pi/8) or
                (dw <= -np.pi/8 and dw > -np.pi/2)):
          # SR = max(S, SR)
          SR += S

    return SL, SR
    
    
  def RobotUpdate(self, x1, y1, w1, UL, UR, dt):
    """
    Updates the position (x1,y1) and orientation w1 of the robot given
    wheel velocities UL (left) and UR (right), where dt is the step size.

    Outputs:
    x2, y2, w2 -- New position and orientation of the robot.
    """

    A = 1  # axle length

    B  = (UL + UR)/2.0
    C  = UR - UL
    dx = B*np.cos(w1)
    dy = B*np.sin(w1)
    dw = np.arctan2(C, A)

    x2 = x1 + dt*dx
    y2 = y1 + dt*dy
    w2 = w1 + dt*dw

    w2 = np.mod(w2+np.pi, 2*np.pi) - np.pi
    if w2 < 0:
      w2 = 2*np.pi + w2

    if x2 > self.xmax:
      x2 = x2 - self.xmax
    if y2 > self.ymax:
      y2 = y2 - self.ymax
    if x2 < 0:
      x2 = self.xmax + x2
    if y2 < 0:
      y2 = self.ymax + y2

    return x2, y2, w2

