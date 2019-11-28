import time
import numpy as np
from feedback import Valence
import tile_coding as tc
import random
import action
import util
from IPython import embed
import time

class DynaQLearner(object):
    def __init__(self):
        self.actionSet = action.ActionSet

        self.dynaQUpdates = 20

        self.alpha = 0.5 # learning rate
        self.gamma = 0.8 # future discount

        # Set up the tiling based on a standard random seed
        random.seed()
        self.numTilings = 4 #8
        self.numDimensions = 2#6
        maxVal = 100
        vals = [0 for i in range(self.numDimensions)]
        self.iht = tc.IHT(maxVal)

        # Initialize with 0s to standardize randomness of hashing across experiments
        tc.tiles(self.iht, self.numTilings, vals)

        # We need to store the s-a value where our state is the 8d tiling
        #  and action is a change in acceleration

        # we WISH we could do this in an np array, but it would be a 1024^8 size array (where each entry was the action values)
        # self.Q = np.zeros([self.numTilings, len(self.actionSet)])
        # Maybe a dictionary will work?
        # self.Q = {}
        # self.StateReward = {}
        # self.TilingToState = {}

        # Currently a full Q array is ~2.4gb
        self.Q = self.setupQ(maxVal)
        self.model = self.setupModel(maxVal)
        # embed()

    def setupQ(self, maxVal):
        entries = [maxVal for i in range(self.numTilings)]
        entries.append(len(action.ActionSet))
        Q = np.zeros(entries)
        return Q
    
    def setupModel(self, maxVal):
        # same setup as Q, but instead of the action setup its a list thats [numTilings, reward]
        entries = [maxVal for i in range(self.numTilings)]
        entries.append(len(action.ActionSet)) # the number of actions we can take
        entries.append(self.numTilings + 1) # +1 for reward
        model = np.zeros(entries)
        return model

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

    # Wrapper to get the tile representation of a state
    def tileRepresentation(self, state):
        return self.getTile(state)

    def update(self, s, a_idx, s_prime, r):
        s_tile = self.getTile(s)
        s_prime_tile = self.getTile(s_prime)

        entry = s_tile
        entry.append(a_idx)
        s_a_entry = tuple(entry)
        q0 = self.Q[s_a_entry]
        q1 = np.max(self.Q[tuple(s_prime_tile)])

        self.Q[tuple(entry)] = q0 + self.alpha * (r + self.gamma * q1 - q0)

        # embed()
        modelEntry = s_prime_tile
        modelEntry.append(r)
        self.model[s_a_entry] = modelEntry
        # embed()
        
        # Saved for hashing tiles
        # h_s = self.getTiledStateHash(self.getTile(s))
        # h_s_prime = self.getTiledStateHash(self.getTile(s_prime))

        # if (h_s not in self.Q):
        #     self.Q[h_s] = action.blankActionValueSet()
        #     self.TilingToState[h_s] = s
        #     self.StateReward[h_s] = action.blankActionValueSet()

        # self.StateReward[h_s][a_idx] = r

        # if (h_s_prime not in self.Q):
        #     self.Q[h_s_prime] = action.blankActionValueSet()
        #     self.TilingToState[h_s_prime] = s
        #     self.StateReward[h_s_prime] = action.blankActionValueSet()

        # self.Q[h_s][a_idx] = self.Q[h_s][a_idx] + self.alpha * (r + self.gamma * np.max(self.Q[h_s_prime]) - self.Q[h_s][a_idx])
        
    def acceptFeedback(self, state):
        pass

    def sampleAction(self, state):
        actionIdx = None
        if (random.random() > 0.9):
            # Select random action
            actionIdx = action.getRandomActionIdx()
        else:
            tile = self.tileRepresentation(state)
            # embed()
            actionIdx = util.randomArgMax(self.Q[tuple(tile)])
            # state_tiled = self.getTile(state)
            # state_tiled_hash = self.getTiledStateHash(state_tiled)
            # if state_tiled_hash in self.Q:
            #     actionIdx = util.randomArgMax(self.Q[state_tiled_hash])
                # embed()
            # else:
            #     self.Q[state_tiled_hash] = action.blankActionValueSet()
            #     self.TilingToState[state_tiled_hash] = state
            #     self.StateReward[state_tiled_hash] = action.blankActionValueSet()
            #     actionIdx = action.getRandomActionIdx()

        # print("sampled action ", actionIdx)
        return actionIdx

    def modelUpdate(self):
        for i in range(self.dynaQUpdates):
            # embed()
            start_state_tiling_hash = random.choice(list(self.Q.keys()))
            if (not start_state_tiling_hash in self.TilingToState):
                embed()

            start_state = list(self.TilingToState[start_state_tiling_hash])
            actionIdx = action.getRandomActionIdx()

            a = action.ActionSet[actionIdx]

            dt = 0.1
            s_prime = [start_state[i] + (a[i] * dt) for i in range(len(start_state))]

            # # update accelerations
            # s_prime[4] += a[0] * dt
            # s_prime[5] += a[1] * dt

            # # update velocities
            # s_prime[2] += s_prime[4] * dt
            # s_prime[3] += s_prime[5] * dt

            # # update positions
            # s_prime[0] += s_prime[2] * dt
            # s_prime[1] += s_prime[3] * dt

            self.update(start_state, actionIdx, s_prime, self.StateReward[start_state_tiling_hash][actionIdx])



    



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