import time
import numpy as np
from feedback import Valence
import tile_coding as tc
import random
import action
import util
from IPython import embed

class DynaQLearner(object):
    def __init__(self):
        self.actionSet = action.ActionSet

        # Set up the tiling based on a standard random seed
        random.seed()
        self.numTilings = 8
        self.numDimensions = 6
        maxVal = 1024
        vals = [0 for i in range(6)]
        self.iht = tc.IHT(maxVal)

        # Initialize with 0s to standardize randomness of hashing across experiments
        tc.tiles(self.iht, self.numTilings, vals)

        # We need to store the s-a value where our state is the 8d tiling
        #  and action is a change in acceleration

        # we WISH we could do this in an np array, but it would be a 1024^8 size array (where each entry was the action values)
        # self.Q = np.zeros([self.numTilings, len(self.actionSet)])
        # Maybe a dictionary will work?
        self.Q = {}

        # embed()

    def getTile(self, state):
        if (len(state) != self.numDimensions):
            print("ERROR: unexpected state size for tiling")
            return None
        else:
            return tc.tiles(self.iht, self.numTilings, state)

    def getTiledStateHash(self, tiledState):
        h = ""
        for entry in tiledState:
            h += str(entry) + "-"
        return h

    def update(self, s, a_idx, s_prime, r):
        h_s = self.getTiledStateHash(self.getTile(s))
        h_s_prime = self.getTiledStateHash(self.getTile(s_prime))

        if (h_s not in self.Q):
            self.Q[h_s] = action.blankActionValueSet()

        if (h_s_prime not in self.Q):
            self.Q[h_s_prime] = action.blankActionValueSet()

        self.Q[h_s][a_idx] = self.Q[h_s][a_idx] + 0.8 * (r + 0.5 * np.max(self.Q[h_s_prime]) - self.Q[h_s][a_idx])
        
    def acceptFeedback(self, state):
        pass

    def sampleAction(self, state):
        actionIdx = None
        if (random.random() > 0.9):
            # Select random action
            actionIdx = action.getRandomActionIdx()
        else:
            state_tiled = self.getTile(state)
            state_tiled_hash = self.getTiledStateHash(state_tiled)
            if state_tiled_hash in self.Q:
                actionIdx = util.randomArgMax(self.Q[state_tiled_hash])
                # embed()
            else:
                self.Q[state_tiled_hash] = action.blankActionValueSet()
                actionIdx = action.getRandomActionIdx()

            
        return actionIdx

    



if __name__ == '__main__':
    DynaQLearner()

    # Make the hash handler
    random.seed(0)
    iht = tc.IHT(1024)
    numTilings = 8
    vals = [0,0,0,0,0,0]
    (p0, p1, v0, v1, a0, a1) = vals
    for i in range(1000):
        vals[0] += random.random() * 2 * np.pi
        vals[1] += random.random() * 2 * np.pi
        vals[2] += random.random() * 2
        vals[3] += random.random() * 2
        vals[4] += random.random() * 0.5
        vals[5] += random.random() * 0.5

        print(tc.tiles(iht, numTilings, vals))