import kinematics_starter
from math import degrees, pi

import numpy as np
import warnings
warnings.filterwarnings("ignore")

robot = kinematics_starter.robot()

def test_one_cycle():
    theta = np.array([0.2,0.3,0.2,1.0,0.5,0.1])
    theta_deg = [degrees(x) for x in theta]
    fk = robot.getFK(theta_deg) # FK accepts input in degrees

    ik = np.array(robot.getIK(fk))

    error = np.sqrt(np.mean((ik-theta)**2))
    
    assert error < 0.01

def test_no_ik():
    theta = np.array([0.2,0.3,0.2,1.0,0.5,0.1])
    theta_deg = [degrees(x) for x in theta]
    fk = robot.getFK(theta_deg) # FK accepts input in degrees

    fk[0,3] = 10
    ik = np.array(robot.getIK(fk))

    assert ik==None