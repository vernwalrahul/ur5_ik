'''
Kinematics Challenge
'''

import numpy as np
import math3d as m3d
from math import *
import copy

class robot(object):
    '''
    robot is a class for kinematics and control for a robot arm
    '''

    def __init__(self, robot_type = 'UR5', base_frame=m3d.Transform, tool_transform=m3d.Transform):
        self.base_frame = base_frame
        self.tool_transform = tool_transform

        assert robot_type == 'UR5', "No dh parameters available for "+str(robot_type)

        # DH parameters. Reference: https://www.universal-robots.com/how-tos-and-faqs/faq/ur-faq/parameters-for-calculations-of-kinematics-and-dynamics-45257/
        # d (unit: mm)
        d1 = 0.089159 
        d2 = d3 = 0
        d4 = 0.10915
        d5 = 0.09465
        d6 = 0.0823

        # a (unit: mm)
        a1 = a4 = a5 = a6 = 0
        a2 = -0.425
        a3 = -0.39225

        # List type of D-H parameter
        self.d = np.array([d1, d2, d3, d4, d5, d6]) # unit: mm
        self.a = np.array([a1, a2, a3, a4, a5, a6]) # unit: mm
        self.alpha = np.array([pi/2, 0, 0, pi/2, -pi/2, 0]) # unit: radian

        # Robot target transformation
        self.target_transform = np.identity(4)

        # Stopping IK calculation flag
        self.stop_flag = False

        # Robot joint solutions data
        self.theta1 = np.zeros(2)
        self.flags1 = None
        self.theta5 = np.zeros((2,2))
        self.flags5 = None
        self.theta6 = np.zeros((2,2))
        self.theta2 = np.zeros((2,2,2))
        self.theta3 = np.zeros((2,2,2))
        self.flags3 = None
        self.theta4 = np.zeros((2,2,2))
        self.debug = False

    def invTransform(self, Transform):
        T = Transform
        R = T[0:3,0:3]
        t = T[0:3,3]

        inverseT = np.concatenate((R.transpose(),-R.transpose().dot(t).reshape(3,1)), axis = 1)
        inverseT = np.vstack((inverseT,[0,0,0,1]))
        return np.asarray(inverseT)

    def link_transform(self, a, d, alpha, theta):
        T = np.array([
                [cos(theta),-sin(theta)*cos(alpha),sin(theta)*sin(alpha) ,a*cos(theta)],
                [sin(theta),cos(theta)*cos(alpha) ,-cos(theta)*sin(alpha),a*sin(theta)],
                [0       ,sin(alpha)          ,cos(alpha)          ,d         ],
                [0       ,0                  ,0                  ,1          ]
            ])
        return T

    def normalize(self,value):
        # This function will normalize the joint values according to the joint limit parameters
        normalized = value
        while normalized > pi:
            normalized -= 2 * pi
        while normalized < -pi:
            normalized += 2* pi
        return normalized

    def select(self, q_sols, q_d, w=[1]*6):
        """Select the optimal solutions among a set of feasible joint value 
        solutions.

        input:
            q_sols: A set of feasible joint value solutions (unit: radian)
            q_d: A list of desired joint value solution (unit: radian)
            w: A list of weight corresponding to robot joints
        Returns:
            A list of optimal joint value solution.
        """

        error = []
        for q in q_sols:
            error.append(sum([w[i] * (q[i] - q_d[i]) ** 2 for i in range(6)]))
        
        return q_sols[error.index(min(error))]


    def getFK(self, joint_angles):
        ''' To calculate the forward kinematics input joint_angles which is a list or numpy array of 6 joint angles

            input = joint angles in degrees
        '''

        T_06 = np.eye(4)

        # convert joint angles to radian
        joint_angles = [radians(i) for i in joint_angles]
        
        for i in range(6):
            T_06 = T_06.dot(self.link_transform(self.a[i], self.d[i], self.alpha[i],joint_angles[i]))
        return T_06

    def getFlags(self,nominator,denominator):
        # check whether the joint value will be valid or not
        if denominator == 0:
            return False
        # tolerance of 0.01
        return abs(nominator/denominator) < 1.01

    def getTheta1(self):
        self.flags1 = np.ones(2)

        t1 = self.target_transform.dot(np.array([0,0,-self.d[5],1]))
        p05 = t1-np.array([0,0,0,1])
        phi1 = atan2(p05[1],p05[0])

        L = sqrt(p05[0]**2+p05[1]**2)

        if abs(self.d[3]) > L:
            self.flags1[:] = self.getFlags(self.d[3],L) # false if the ratio > 1.001
            L = abs(self.d[3])
        phi2 = acos(self.d[3]/L)

        self.theta1[0] = self.normalize(phi1+phi2+pi/2)
        self.theta1[1] = self.normalize(phi1-phi2+pi/2);
        if self.debug:
            print('t1: ', self.theta1)
            print('flags1: ',self.flags1)
        # stop the program early if no solution is possible
        self.stop_flag = not np.any(self.flags1)
    
    def getTheta5(self):
        # This function will solve joint 5
        self.flags5 = np.ones((2,2))

        p06 = self.target_transform[0:3,3]
        for i in range(2):
            p16z = p06[0]*sin(self.theta1[i])-p06[1]*cos(self.theta1[i]);
            L = self.d[5]

            if abs(p16z - self.d[3]) > L:
                self.flags5[i,:] = self.getFlags(p16z - self.d[3],self.d[5])
                L = abs(p16z-self.d[3]);
            theta5i = acos((p16z-self.d[3])/L)
            self.theta5[i,0] = theta5i
            self.theta5[i,1] = -theta5i

        if self.debug:
            print('t5: ', self.theta5)
            print('flags5: ',self.flags5)

        # stop the program early if no solution is possible
        self.stop_flag = not np.any(self.flags5)

    def getTheta6(self):
        # This function will solve joint 6
        for i in range(2):
            T1 = self.link_transform(self.a[0],self.d[0],self.alpha[0],self.theta1[i])
            T61 = self.invTransform(self.invTransform(T1).dot(self.target_transform))
            for j in range(2):
                if sin(self.theta5[i,j]) == 0:
                    if self.debug:
                        print("Singular case. selected theta 6 = 0")
                    self.theta6[i,j] = 0
                else:
                    self.theta6[i,j] = atan2(-T61[1,2]/sin(self.theta5[i,j]),
                                              T61[0,2]/sin(self.theta5[i,j]))
        
        if self.debug:
            print('t6: ', self.theta6)

    def getTheta23(self):
        # This function will solve joint 2 and 3
        self.flags3 = np.ones ((2,2,2))
        for i in range(2):
            T1 = self.link_transform(self.a[0],self.d[0],self.alpha[0],self.theta1[i])
            T16 = self.invTransform(T1).dot(self.target_transform)
            
            for j in range(2):
                T45 = self.link_transform(self.a[4],self.d[4],self.alpha[4],self.theta5[i,j])
                T56 = self.link_transform(self.a[5],self.d[5],self.alpha[5],self.theta6[i,j])
                T14 = T16.dot(self.invTransform(T45.dot(T56)))

                P13 = T14.dot(np.array([0,-self.d[3],0,1]))-np.array([0,0,0,1])
                L = P13.dot(P13.transpose()) - self.a[1]**2 - self.a[2]**2

                if abs(L / (2*self.a[1]*self.a[2]) ) > 1:
                    self.flags3[i,j,:] = self.getFlags(L,2*self.a[1]*self.a[2])
                    L = np.sign(L) * 2*self.a[1]*self.a[2]
                self.theta3[i,j,0] = acos(L / (2*self.a[1]*self.a[2]) )
                self.theta2[i,j,0] = -atan2(P13[1],-P13[0]) + asin( self.a[2]*sin(self.theta3[i,j,0])/np.linalg.norm(P13) )
                self.theta3[i,j,1] = -self.theta3[i,j,0]
                self.theta2[i,j,1] = -atan2(P13[1],-P13[0]) + asin( self.a[2]*sin(self.theta3[i,j,1])/np.linalg.norm(P13) )

        if self.debug:
            print('t2: ', self.theta2)
            print('t3: ', self.theta3)
            print('flags3: ',self.flags3)

        # stop the program early if no solution is possible
        self.stop_flag = not np.any(self.flags3)
    
    def getTheta4(self):
        # This function will solve joint 4 value
        for i in range(2):
            T1 = self.link_transform(self.a[0],self.d[0],self.alpha[0],self.theta1[i])
            T16 = self.invTransform(T1).dot(self.target_transform)
            
            for j in range(2):
                T45 = self.link_transform(self.a[4],self.d[4],self.alpha[4],self.theta5[i,j])
                T56 = self.link_transform(self.a[5],self.d[5],self.alpha[5],self.theta6[i,j])
                T14 = T16.dot(self.invTransform(T45.dot(T56)))

                for k in range(2):
                    T13 = self.link_transform(self.a[1],self.d[1],self.alpha[1],self.theta2[i,j,k]).dot(
                          self.link_transform(self.a[2],self.d[2],self.alpha[2],self.theta3[i,j,k]) )
                    T34 = self.invTransform(T13).dot(T14)
                    self.theta4[i,j,k] = atan2(T34[1,0],T34[0,0])

        if self.debug:
            print('t4: ', self.theta4)

    def countValidSolution(self):
        # This function will count the number of available valid solutions
        number_of_solution = 0
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    if self.flags1[i] and self.flags3[i,j,k] and self.flags5[i,j]:
                        number_of_solution += 1
        return number_of_solution

    def getSolution(self):
        # This function will call all function to get all of the joint solutions
        for i in range(4):
            if i == 0:
                self.getTheta1()
            elif i == 1:
                self.getTheta5()
            elif i == 2:
                self.getTheta6()
                self.getTheta23()
            elif i == 3:
                self.getTheta4()

            # This will stop the solving the IK when there is no valid solution from previous joint calculation
            if self.stop_flag:
                return

    def getIK(self, ee_TM, seed_joint_angles=np.zeros((6,))):
        ''' Analytically solve the inverse kinematics

            inputs: ee_TM (4X4 array) = end effector transformation matrix

                    seed_joint_angles = [theta1, theta2, theta3, theta4, theta5, theta6]
                                         seed angles for comparing solutions to ensure mode change does not occur

            outputs: joint_angles = joint angles in rad to reach goal end-effector pose
        '''
        # self.target_transform = self.pose_to_matrix(poseGoal)
        self.target_transform = ee_TM

        if self.debug:
            print('Input to IK:\n', self.target_transform)

        self.getSolution()
        number_of_solution = self.countValidSolution()
        if self.stop_flag or number_of_solution < 1:
            if self.debug:
                print('No solution')
            return None

        Q = np.zeros((number_of_solution,6))
        index = 0
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    if not (self.flags1[i] and self.flags3[i,j,k] and self.flags5[i,j]):
                        # skip invalid solution
                        continue
                    Q[index,0] = self.normalize(self.theta1[i])
                    Q[index,1] = self.normalize(self.theta2[i,j,k])
                    Q[index,2] = self.normalize(self.theta3[i,j,k])
                    Q[index,3] = self.normalize(self.theta4[i,j,k])
                    Q[index,4] = self.normalize(self.theta5[i,j])
                    Q[index,5] = self.normalize(self.theta6[i,j])
                    index += 1

        return self.select(Q, seed_joint_angles)

    def motionPlanner(self, phases=np.array([0,.25,.75,1]), poses=np.zeros((4,6))):
        timed_poses = np.zeros((100,7))
        # YOUR CODE GOES HERE


        return timed_poses



    def xyzToolAngleError(self, joint_angles, poseGoal, angleErrorScale=.02):
        ''' Calculate the error between the goal position and orientation and the actual
            position and orientation

            inputs: poseGoal = [x,y,z,rx,ry,rz] position of goal. orientation of goal specified in rotation vector (axis angle) form [rad]
                    joint_angles = current joint angles from optimizer [rad]
                    angleErrorScale = coefficient to determine how much to weight orientation in
                                      the optimization. value of ~ .05 has equal weight to position
        '''

        ####  YOUR CODE GOES HERE
        totalError = 0

        return totalError


    def getIKnum(self, xyzGoal, eulerAnglesGoal, seed_joint_angles=np.zeros((6,))):
        ''' Numerically calculate the inverse kinematics through constrained optimization

            inputs: poseGoal = [x,y,z,rx,ry,rz] goal position of end-effector in global frame.
                                Orientation [rx,ry,rz] specified in rotation vector (axis angle) form [rad]
                    seed_joint_angles = [theta1, theta2, theta3, theta4, theta5, theta6]
                                         seed angles for comparing solutions to ensure mode change does not occur

            outputs: joint_angles = joint angles in rad to reach goal end-effector pose
        '''
        joint_angles = np.zeros((1,6))
        ####  YOUR CODE GOES HERE

        return joint_angles
