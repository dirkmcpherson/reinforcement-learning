'''
Author: James Staley
email: james.staley625703@tufts.edu
March 2020 :(
'''

from IPython import embed
from FA_config import *
import numpy as np
import agent_utils as u
import random

# Import cupsworld to grab relevant dimensions
# import sys
# sys.path.append('~/workspace/cupsworld/env/')
# import cupsworld

WIDTH = 800
HEIGHT = 600

# GRIPPER = "gripper"
# POS = "pos"

'''
StateHandler. Just to simplify the rest of the code, this takes our state from the environment and reformulates it so its useful 
'''
class StateHandler(object):
    def __init__(self):
        self.gripperState = False
        self.gripperPose = (-1,-1)

        self.numStates = None
        self.hasPickedUp = False
        self.raisedObject = False
    # def parseCupsworldJSON(self, jsonObject):
        # {"Grippers": [{"id": 4, "pos": [400, 300], "status": false}], "Objects": [{"id": 1, "name": "block", "pos": [399, 94]}, {"id": 2, "name": "block", "pos": [199, 94]}, {"id": 3, "name": "cup", "pos": [266, 115]}], "Action": {"name": "", "result": ""}}

    def reset(self):
        self.hasPickedUp = False

    def formatCupsworldState(self, jsonObject):
        state = None
        # if (WORLD_RELATIVE_TO_GRIPPER):
        #     state = self.formatCupsworldState_world_gripperRelative(jsonObject)
        # elif (PICKUP_AGENT):
        #     state = self.formatCupsworldState_pickupAgent(jsonObject)
        # elif (GOTO_AGENT):
        #     state = self.formatCupsworldState_pickupAgent(jsonObject)
        state = self.formatCupsworldState_debugState(jsonObject)

        if (self.numStates is None):
            self.numStates = len(state)
            print("NumStates: ", self.numStates)

        return state

    def formatCupsworldState_world_gripperRelative(self, jsonObject):
        gripper = jsonObject["Grippers"][0]
        gripperPos = np.array(gripper["pos"])
        gripperClosed = gripper["status"]

        self.gripperPose = gripperPos
        self.gripperState = gripperClosed

        objects = jsonObject["Objects"]
        
        objectPositions = None
        objectPositions = [gripperPos - np.array(entry["pos"]) for entry in objects]

        # Normalize positions to range (0,1)
        objectPositions = [[u.normalizeValue(entry[0], 0, WIDTH), u.normalizeValue(entry[1], 0, HEIGHT)] for entry in objectPositions]

        ret = [1.] # Allow for representation of affine functions
        [ret.extend(entry) for entry in objectPositions]

        if (ONE_HOT_GRIPPER_STATE):
            ret.extend([0, 1] if gripperClosed else [1, 0])
        else:
            ret.append(1 if gripperClosed else 0)

        if (USE_POLYNOMIAL_FEATURES):
            factor = 1 if gripperClosed else 0
            for entry in objectPositions:
                ret.append(entry[0]*factor)
                ret.append(entry[1]*factor)

        return ret

    def formatCupsworldState_debugState(self, jsonObject):
        ret = [1.] # allow for affine transformations

        gripper = jsonObject["Grippers"][0]
        gripperPos = np.array(gripper["pos"])
        gripperClosed = gripper["status"]
        objects = jsonObject["Objects"]
        
        self.gripperPose = gripperPos
        self.gripperState = gripperClosed
        gripped = 1 if gripperClosed else 0

        # normalized gripper state
        gripper_x_norm = u.normalizeValue(self.gripperPose[0], 0, WIDTH)
        gripper_y_norm = u.normalizeValue(self.gripperPose[1], 0, HEIGHT)

        objectPositions = [np.array(entry["pos"]) for entry in objects]
        # Theta and distance to objects, relative to gripper

        distances = []
        # for p in objectPositions:
        #     theta = u.angle_between(self.gripperPose, p)
        #     theta = u.normalizeValue(theta, -np.pi, np.pi)
        #     distance = u.distance(self.gripperPose, p)
        #     distance = u.normalizeValue(distance, 0, WIDTH)
        #     if distance > 1.:
        #         distance = 1.

        #     ret.append(theta)
        #     ret.append(distance)
        
        p = objectPositions[1]
        theta = u.angle_between(self.gripperPose, p)
        theta = u.normalizeValue(theta, -np.pi, np.pi)
        distance = u.distance(self.gripperPose, p)
        distance = u.normalizeValue(distance, 0, WIDTH)
        if distance > 1.:
            distance = 1.

        # ret.append(theta)
        ret.append(distance)
        return ret

    def formatCupsworldState_pickupAgent(self, jsonObject, objectOfInterest=False):
        ret = [1.] # allow for affine transformations

        gripper = jsonObject["Grippers"][0]
        gripperPos = np.array(gripper["pos"])
        gripperClosed = gripper["status"]
        objects = jsonObject["Objects"]
        
        self.gripperPose = gripperPos
        self.gripperState = gripperClosed
        gripped = 1 if gripperClosed else 0

        # normalized gripper state
        gripper_x_norm = u.normalizeValue(self.gripperPose[0], 0, WIDTH)
        gripper_y_norm = u.normalizeValue(self.gripperPose[1], 0, HEIGHT)

        objectPositions = [np.array(entry["pos"]) for entry in objects]
        # Theta and distance to objects, relative to gripper

        distances = []
        # for p in objectPositions:
        #     theta = u.angle_between(self.gripperPose, p)
        #     theta = u.normalizeValue(theta, -np.pi, np.pi)
        #     distance = u.distance(self.gripperPose, p)
        #     distance = u.normalizeValue(distance, 0, WIDTH)
        #     if distance > 1.:
        #         distance = 1.

        #     ret.append(theta)
        #     ret.append(distance)
        
        p = objectPositions[1]
        theta = u.angle_between(self.gripperPose, p)
        theta = u.normalizeValue(theta, -np.pi, np.pi)
        distance = u.distance(self.gripperPose, p)
        distance = u.normalizeValue(distance, 0, WIDTH)
        if distance > 1.:
            distance = 1.

        # ret.append(theta)
        ret.append(distance)
        # ret.append(theta*distance)
        # ret.append(theta**2)
        # ret.append(distance**2)
        # ret.append(theta*distance**2)
        # ret.append(theta**2 * distance)
        # ret.append(theta**2 * distance**2)

        # ret.append(gripped*distance)

        #     distances.append(distance)
        # ret.append(gripper_x_norm*theta)
        # ret.append(gripper_y_norm*theta)
        # ret.append(gripper_x_norm*distance)
        # ret.append(gripper_y_norm*distance)

        # self.raisedObject = False
        # if (distance < .2 and gripper_y_norm > PICKUP_AGENT_HEIGHT):
        #     self.raisedObject = True
        # for d in distances:
        #     if (distance < .2 and gripper_y_norm > 0.4):
        #         self.raisedObject = True
        
        # ret.append(gripper_x_norm)
        # ret.append(gripper_y_norm)
        # ret.append(gripped*gripper_x_norm)
        # ret.append(gripped*gripper_y_norm)
        # ret.append(gripped)

        return ret

    def breakDueToConstraints(self):
        if (self.gripperPose[0] < 0 or self.gripperPose[1] > WIDTH or self.gripperPose[1] < 0 or self.gripperPose[1] > HEIGHT):
            return True

    def rewardCupsworld_pickupAgent(self, state, previousState):
        gripperState = state[-1]
        previousGripperState = previousState[-1]
        gripper_y_norm_gripped = state[-2]

        if (not self.hasPickedUp and gripperState and not previousGripperState):
            self.hasPickedUp = True
            return 10
        elif ((gripper_y_norm_gripped) >= PICKUP_AGENT_HEIGHT or self.raisedObject):
            return 100
        else:
            return DEFAULT_REWARD if (not self.gripperState) else (gripper_y_norm_gripped) # gripper_y_norm

    def rewardCupsworld_gotoAgent(self, state):
        done = self.terminalCupsworldState_gotoAgent(state)
        reward = DEFAULT_REWARD

        if (done):
            reward = 100
        
        # print("{:2.2f}, {:2.2f}, {}".format(state[-4], state[-1], reward))
        return reward, done

    def terminalCupsworldState_pickupAgent(self, state):
        return (state[-2] >= PICKUP_AGENT_HEIGHT)

    def terminalCupsworldState_gotoAgent(self, state):
        return (state[1] <= GOTO_AGENT_DISTANCE) and self.gripperState

    def terminalCupsworldState(self, state):
        if (PICKUP_AGENT):
            return self.terminalCupsworldState_pickupAgent(state)
        elif (GOTO_AGENT):
            return self.terminalCupsworldState_gotoAgent

        return self.gripperState and self.gripperPose[0] > 350 # Gripper closed is true

    def getCupsworldReward(self, state, previousState):
        if (PICKUP_AGENT):
            return self.rewardCupsworld_pickupAgent(state, previousState)
        elif (GOTO_AGENT):
            return self.rewardCupsworld_gotoAgent(state)

        if (self.terminalCupsworldState(state)):
            return 100
        elif (self.gripperState):
            return 0
        else: 
            return -1
            # return 3*1400 - (sum(state[:6])/(3*1400.))


    # Given a list of features, s, where each feature is in the range [0,1]. Return a set of fourier basis functions
    def convertToFourierBasis(self, s):
        if (np.any(s < 0) or np.any(s > 1)):
            print("Incompatible values for fourier basis. Must be in range [0,1]: ", s)
            return None

import itertools
class blah():
    def __init__(self, nvars, order=3):
        nterms = pow(order + 1.0, nvars)
        self.numTerms = nterms
        self.order = order
        # self.ranges = np.array(ranges)
        iter = itertools.product(range(order+1), repeat=nvars)
        self.multipliers = np.array([list(map(int,x)) for x in iter])

    def computeFeatures(self, features):
        if len(features) == 0:
            return np.ones((1,))
        basisFeatures = features #np.array([self.scale(features[i],i) for i in range(len(features))])

        embed()
        return np.cos(np.pi * np.dot(self.multipliers, basisFeatures))


if __name__ == "__main__":
    a = blah(10)

    s = [random.random() for i in range(10)]

    print(a.computeFeatures(s))