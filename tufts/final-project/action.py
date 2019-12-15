import random
import numpy as np

increment = round(np.pi / 2., 2)
ActionSet = {
            # 0: [-0.1, 0],
            # 1: [0.1, 0]
            # 0: [0., 0.],
            # 0: [0, increment],
            # 1: [0, -increment]
            0: [increment, 0.],
            1: [-increment, 0.],
            2: [0., increment],
            3: [0., -increment]
        }

def getRandomActionIdx():
    return random.randint(0, (len(ActionSet)-1))

def getRandomAction():
    return ActionSet[getRandomActionIdx()]

def blankActionValueSet():
    # blank = {}
    # for i in range(len(ActionSet)):
    #     blank[i] = 0.

    # return blank
    return [0 for entry in ActionSet]

