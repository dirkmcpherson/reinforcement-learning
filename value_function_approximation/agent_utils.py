'''
Author: James Staley
email: james.staley625703@tufts.edu
March 2020 :(
'''

import math
import numpy as np

# Normalize a value to [0,1]
# NO CHECK IS DONE TO DETERMINE IF VALUE LIES WITHIN RANGE
def normalizeValue(value, minVal, maxVal):
    remapped = float(value - minVal) / (maxVal - minVal)
    # print("Remapped %f to %f" % (value, remapped))
    return remapped

def angle_between(p0, p1):
    return math.atan2(p1[1] - p0[1], p1[0] - p0[0])

def distance(p0, p1):
    return np.sqrt( (p0[0] - p1[0])**2 + (p0[1] - p1[1])**2 )

if __name__ == "__main__":
    pass