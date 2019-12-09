import random
import numpy as np

def randomArgMax(choices):
    return random.choice( np.argwhere(choices == np.max(choices)).flatten().tolist() )


def angleToState(angle):
    return int(np.ceil(angle * 100))    