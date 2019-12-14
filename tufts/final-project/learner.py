import time
import numpy as np
from feedback import Valence
import tile_coding as tc
import random
import action
import util
from IPython import embed
import time
from history import History

pathForTable = "./q_values"
pathForModel = "./model"

class DynaQLearner(object):
    def __init__(self):
        self.actionSet = action.ActionSet

        self.dynaQUpdates = 50

        self.alpha = 1 #0.5 # learning rate
        self.gamma = 0.9 # future discount

        # Set up the tiling based on a standard random seed
        random.seed()
        self.numTilings = 3 #8
        self.numDimensions = 2#6
        maxVal = 1000
        vals = [0 for i in range(self.numDimensions)]
        self.iht = tc.IHT(maxVal)

        # Initialize with 0s to standardize randomness of hashing across experiments
        tc.tiles(self.iht, self.numTilings, vals)

        self.history = History()
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
        # self.visitedStates = dict()

        # self.visitedStatesCount = dict()
        # self.visitedStatesModelCount = dict()

        self.normFirst = 0
        self.dynaFirst = 0

    def setupQ(self, maxVal):
        entries = [maxVal for i in range(self.numTilings)]
        entries.append(len(action.ActionSet))
        Q = np.zeros(entries)
        # resolution = int(np.ceil( round((2*np.pi) * 100,2) ))
        # entries = [resolution]
        # entries.append(len(action.ActionSet))
        # Q = np.zeros(entries)
        return Q
    
    # In order to space, we're going to store (s,a,r) rather than (s,a,s',r) since we dont actually have to use s'
    def setupModel(self, maxVal):
        # same setup as Q, but instead of the action setup its a list thats [numTilings, reward]
        # resolution = int(np.ceil( (2*np.pi) / 0.01 ))
        # entries = [resolution]
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
        # return util.angleToState(round(state[0], 2))
        return self.getTile(state)

    def update(self, s, a_idx, s_prime, r, updateModel = False):
        # update out history if we're not doing a modelUpdate
        if (updateModel):
            self.history.push((s, a_idx, s_prime))

        s_tile = self.tileRepresentation(s)
        entry = s_tile
            # print("{} took action {} to {} for reward {}".format(s_tile,a_idx,s_prime,r))

        s_prime_tile = self.tileRepresentation(s_prime)

        entry.append(a_idx)
        s_a_entry = tuple(entry)

        # if not updateModel:
        #     embed()

        q0 = self.Q[s_a_entry]
        q1 = np.max(self.Q[tuple(s_prime_tile)])
        # Cap the final value because right now it can just accumulate forever (there are several terminal states and termination requires resting on them, so we can't just end as soon as we reach the square)
        
        updated = q0 + self.alpha * (r + self.gamma * q1 - q0)
        # embed()
        # time.sleep(1)
        # if (updated > 10):
            # embed()
        updated = min(10., updated)
        self.Q[s_a_entry] = updated

        # print("{} {} to {} for reward {} and to update {} by {} final value {}".format(("" if updateModel else "MODEL"),s_a_entry,s_prime,r,q0,dv,self.Q[s_a_entry]))
        # print("     q0:{} q1:{} r:{}".format(q0,q1,r))
        if updateModel:
            self.model[s_a_entry] = r
        else:
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
        
    def acceptFeedback(self, window, reward):
        for (s,a_idx,s_p) in window:
            self.update(s, a_idx, s_p, reward)
        return

    def sampleAction(self, state):
        # embed()
        actionIdx = None
        tile = self.tileRepresentation(state)
        if (random.random() > 0.9):
            # Select random action
            actionIdx = action.getRandomActionIdx()
        else:
            actionIdx = util.randomArgMax(self.Q[tuple(tile)])
            # actionIdx = util.randomArgMax(self.Q[tuple(tile)])


        state_str = str([round(entry,2) for entry in state])
        if state_str not in self.visitedStates:
            self.visitedStates.add(state_str)

        # tile_str = str(tile)
        # if (tile_str not in self.visitedStates.keys()):
        #     self.visitedStates[tile_str] = [(state[0], state[1]), actionIdx]
        #     # self.visitedStates[tile] = [actionIdx]
        #     # self.visitedStatesCount[tile] = 0
        # else:
        # #     # embed()
        #     if actionIdx not in self.visitedStates[tile_str]:
        #         self.visitedStates[tile_str].append(actionIdx)

        #     self.visitedStatesCount[tile] += 1

        return actionIdx

    def modelUpdate(self, env):
        for i in range(self.dynaQUpdates):
            # visitedTile = random.choice(list(self.visitedStates.keys()))
            # s = self.visitedStates[visitedTile][0] # state stored as tuple in first entry
            # actionIdx = random.choice(self.visitedStates[visitedTile][1:]) # rest of entry is visited action idx
            
            # t0 = time.time()
            s_str = random.choice(list(self.visitedStates))
            s = s_str.replace(' ','')
            s = s.replace('[','')
            s = s.replace(']','')
            s = s.split(',')
            s = [float(entry) for entry in s]
            # print("   string manipulation took: {}".format(time.time() - t0))
            # t0 = time.time()

            # embed()
            # time.sleep(1)

            actionIdx = action.getRandomActionIdx()
            a = action.ActionSet[actionIdx]

            s_prime = env.update_state_from_action(s, a)
            # print("{} + {} for {}".format(s, a, s_prime))


            # print("--model s{}-{}  a{}  sp{}-{}--".format(s,self.tileRepresentation(s),actionIdx,s_prime,self.tileRepresentation(s_prime)))
            # Grab the reward from our model of the environment
            # s_tile = self.getTile(s)

            # tmp = visitedTile.replace(' ','')
            # tmp = tmp.replace('[','')
            # tmp = tmp.replace(']','')
            # tmp = tmp.split(',')
            # s_a_entry = [int(entry) for entry in tmp]
            # embed()

            s_a_entry = self.tileRepresentation(s)
            s_a_entry.append(actionIdx)
            # embed()
            # time.sleep(1)
            r = self.model[tuple(s_a_entry)]

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

            # embed()
            self.update(s, actionIdx, s_prime, r)
            # print("   everything else took: {}".format(time.time() - t0))
            # t0 = time.time()

        # embed()
    def save(self):
        embed()
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