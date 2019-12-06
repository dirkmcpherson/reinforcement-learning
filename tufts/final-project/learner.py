import time
import numpy as np
from feedback import Valence
import tile_coding as tc
import random
import action
import util
from IPython import embed
import time

pathForTable = "./q_values"
pathForModel = "./model"
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
        self.visitedStates = set()
        # embed()

    def setupQ(self, maxVal):
        entries = [maxVal for i in range(self.numTilings)]
        entries.append(len(action.ActionSet))
        Q = np.zeros(entries)
        return Q
    
    # In order to space, we're going to store (s,a,r) rather than (s,a,s',r) since we dont actually have to use s'
    def setupModel(self, maxVal):
        # same setup as Q, but instead of the action setup its a list thats [numTilings, reward]
        entries = [maxVal for i in range(self.numTilings)]
        entries.append(len(action.ActionSet)) # the number of actions we can take
        entries.append(1) # +1 for reward. We're ignoring S'
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

    def update(self, s, a_idx, s_prime, r, updateModel = False):
        s_tile = self.getTile(s)
        s_prime_tile = self.getTile(s_prime)

        entry = s_tile
        entry.append(a_idx)
        s_a_entry = tuple(entry)
        q0 = self.Q[s_a_entry]
        q1 = np.max(self.Q[tuple(s_prime_tile)])

        self.Q[tuple(entry)] = q0 + self.alpha * (r + self.gamma * q1 - q0)


        if updateModel:
            self.model[s_a_entry] = r
            # print("Updated {} from {} to {}".format(entry, q0, self.Q[tuple(entry)]))
        else:
            # print("Updated value from {} to {}".format(q0, self.Q[tuple(entry)]))
            pass
        
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
        if (state not in self.visitedStates):
            self.visitedStates.add(state)

        return actionIdx

    def modelUpdate(self, env):
        for i in range(self.dynaQUpdates):

            # s = np.array(env.get_random_state())
            s = list(random.sample(self.visitedStates, 1)[0])
            # print("sampled ", s)

            actionIdx = action.getRandomActionIdx()
            a = action.ActionSet[actionIdx]

            s_prime = env.update_state_from_action(s, a)

            # Grab the reward from our model of the environment
            s_tile = self.getTile(s)
            entry = s_tile
            entry.append(actionIdx)
            s_a_entry = tuple(entry)
            r = self.model[s_a_entry]

            # embed()
            # start_state_tiling_hash = random.choice(list(self.Q.keys()))
            # if (not start_state_tiling_hash in self.TilingToState):
            #     embed()

            # start_state = list(self.TilingToState[start_state_tiling_hash])
            # actionIdx = action.getRandomActionIdx()

            # # update accelerations
            # s_prime[4] += a[0] * dt
            # s_prime[5] += a[1] * dt

            # # update velocities
            # s_prime[2] += s_prime[4] * dt
            # s_prime[3] += s_prime[5] * dt

            # # update positions
            # s_prime[0] += s_prime[2] * dt
            # s_prime[1] += s_prime[3] * dt

            self.update(s, actionIdx, s_prime, r)

    def save(self):
        np.save(pathForTable, self.Q)
        np.save(pathForModel, self.model)

    def load(self):
        self.Q = np.load(pathForTable + ".npy")
        self.model = np.load(pathForModel + ".npy")

    



if __name__ == '__main__':
    dql = DynaQLearner()
    t0 = time.time()
    dql.save()
    print("Took {} to save.".format(time.time() - t0))
    t0 = time.time()
    dql.load()
    print("Took {} to load.".format(time.time() - t0))




    # Make the hash handler
    # random.seed(0)
    # iht = tc.IHT(1024)
    # numTilings = 8
    # vals = [0,0,0,0,0,0]
    # (p0, p1, v0, v1, a0, a1) = vals
    # for i in range(1000):
    #     vals[0] += random.random() * 2 * np.pi
    #     vals[1] += random.random() * 2 * np.pi
    #     vals[2] += random.random() * 2
    #     vals[3] += random.random() * 2
    #     vals[4] += random.random() * 0.5
    #     vals[5] += random.random() * 0.5

    #     print(tc.tiles(iht, numTilings, vals))