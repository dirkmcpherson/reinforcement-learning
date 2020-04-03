import numpy as np
import random

# Minecraft state is 5x5 grid where each cell holds a symbol
numSymbols = 10 # The number of possible entrants for a grid space
numGridSpaces = 25
numSensors = 2
numCardinalDirections = 4

maxSensorRange = 30

def generateRandomState():
    ret = []
    for i in range(numGridSpaces):
        hot = random.randint(0,numSymbols)
        for j in range(numSymbols):
            ret.append(1 if j == hot else 0)
    
    for k in range(numSensors*numCardinalDirections):
        ret.append(random.randint(0,maxSensorRange))

    reward = random.randint(0,100)

    return np.array(ret), reward
    

'''
StateFormatter. Just to simplify the rest of the code, this takes our state from the environment and reformulates it so its useful 
'''
class StateFormatter(object):
    def __init__(self):
        pass

    def formatMinecraftState(self, state):
        ret = np.array(numSymbols*numGridSpaces + numSensors*numCardinalDirections)
        # 2 sensors for each cardinal direction indicating the closest "sensorType" in that direction
        
        # Make a one-hot vector for each of the 25 spots
        # TODO: Assumption that state is ordered by rows

        return ret
        

        