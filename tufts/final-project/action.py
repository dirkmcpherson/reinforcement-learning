import random

ActionSet = {
            # 0: [-0.1, 0],
            # 1: [0.1, 0]
            0: [0, 0],
            # 1: [0.5, 0],
            # 2: [-0.5, 0]
            1: [200, 0],
            2: [-200, 0],
            3: [0, 200],
            4: [0, -200]
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

