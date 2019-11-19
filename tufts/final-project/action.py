ActionSet = {
            0: [0, 0],
            1: [0.05, 0],
            2: [-0.05, 0]
        }

def blankActionValueSet():
    blank = {}
    for i in range(len(ActionSet)):
        blank[i] = 0.

    return blank

class Action:
    def __init__(self):
        # The action set constitutes rotational acceleration changes only
        #  One entry per join
        self.ActionSet = {
            0: [0, 0],
            1: [0.05, 0],
            2: [-0.05, 0]
        }

        # constructActionSet()

    # def constructActionSet(self):
    #     dimensions = 2
    #     accelRanges = 0.05



