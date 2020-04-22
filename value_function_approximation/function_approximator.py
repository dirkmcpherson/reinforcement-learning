'''
Author: James Staley
email: james.staley625703@tufts.edu
March 2020 :(
'''

import socket
import time
import random
import numpy as np
import json
from StateHandler import  StateHandler
from logger import Logger
from FA_config import *
from IPython import embed
import matplotlib
# matplotlib.use('QT4Agg') # Use for OSX
import matplotlib.pyplot as plt

'''
Linear Value Function Approximator 
'''
class FA(object):
    def __init__(self, stateHandler, action_set, socket, MANUAL=False, DEBUG=False):
        self.debug = DEBUG
        self.manual = MANUAL
        self.log = Logger(FA_LOG_LEVEL)
        
        # Environment specific
        self.ACTION_SET = action_set
        self.sh = stateHandler
        self.socket = socket
        # Environment specific 

        self.epsilon = 1.0
        self.gamma = FA_GAMMA
        self.alpha = FA_ALPHA

        self.numTerminalStates = 0

        if (self.sh.numStates is None):
            # Get a state sample and figure out what we're working with
            self.sh.getState(self.socket, self.sh)

        self.weightsForActions = dict()
        for action in self.ACTION_SET:
            self.weightsForActions[action] = self.initWeights(np.zeros(self.sh.numStates))
        
        # method specific. an action is taken from a state to get a reward
        self.rewards = []
        self.states = []
        self.actions = []
        self.previousState = None

        self.actionCount = dict()
        for action in ACTION_SET:
            self.actionCount[action] = 0

        self.cumulativeReward = 0
        self.periodicDebugCounter = 0


    def periodicDebug(self, msg):
        if (self.periodicDebugCounter % 100 == 0):
            self.log.info(msg)

        self.periodicDebugCounter += 1

    def reset(self):
        # self.epsilon = 1.
        self.rewards = []
        self.states = []
        self.actions = []
        self.cumulativeReward = 0
        self.stateFormatter.reset()

    def initWeights(self, state):
        # We want a single weight associated with each entry in the state
        w = np.random.rand(len(state))
        return w

    def value(self, state, weights):
        return np.sum(state * weights)

    def gradient(self, state, weights):
        # special case for linear model, x(S) is the features of s, for now, since we have no features its just s
        return state
    
    def epsilonGreedy(self, state):
        if self.epsilon > 0.1:
            self.epsilon -= 0.001

        if (random.random() < self.epsilon):
            return random.choice(ACTION_SET)
        else:
            # modelPredictions = dict()
            # for actionString in ACTION_SET:
            #     modelPredictions[actionString] = self.value(state, self.weightsForActions[actionString])

            # chosenAction = max(modelPredictions.keys(), key=(lambda k: modelPredictions[k]))
            
            # for i, (k,v) in enumerate(modelPredictions.items()):
            #     self.log.trace("{}: {:3.2f}".format(k, v))
            # self.log.trace("Highest Value Action: {}".format(chosenAction))

            # Select action probabilistically based on value
            modelPredictions = []
            mostNeg = 0
            for actionString in ACTION_SET:
                v = self.value(state, self.weightsForActions[actionString])
                if (v < mostNeg):
                    mostNeg = v
                modelPredictions.append([v, actionString])

            
            total = sum([(entry[0]+mostNeg) for entry in modelPredictions])

            for entry in modelPredictions:
                entry[0] = (entry[0]+mostNeg) / total

            modelPredictions.sort(key = lambda k: k[0])
            modelPredictions.reverse()
            
            chosenAction = modelPredictions[0][1]
            choice = random.random()
            cum = 0
            for entry in modelPredictions:
                cum += entry[0]
                if (cum <= choice):
                    chosenAction = entry[1]
                    break

            return chosenAction

    def printActionValue(self, state):
        modelPredictions = dict()
        for actionString in ACTION_SET:
            modelPredictions[actionString] = self.value(state, self.weightsForActions[actionString])

        chosenAction = max(modelPredictions.keys(), key=(lambda k: modelPredictions[k]))
        
        for i, (k,v) in enumerate(modelPredictions.items()):
            self.log.trace("{}: {:3.2f}".format(k, v))
        self.log.trace("Highest Value Action: {}".format(chosenAction))

    def chooseAction(self, state):
        action = None
        if (self.manual):
            self.printActionValue(state)
            action = input("Entry action from {}>>".format(ACTION_SET))
            if (action not in ACTION_SET):
                if (action == "train"):
                    print("Releasing Manual Mode")
                    self.manual = False
                    self.epsilon = 0.25
                    action = self.epsilonGreedy(state)
                else:
                    print("Unrecognized action, randomly selecting.")
                    action = random.choice(ACTION_SET)
        else:
            action = self.epsilonGreedy(state)

        return action

    def stepEnvironment(self, action):
        response = sendMessage(self.socket, action)

        time.sleep(0.05)
        # self.log.debug(("ActionResponse: ", response))

        state = getState(self.socket, self.stateFormatter)
        reward, done = self.stateFormatter.getCupsworldReward(state, self.previousState)

        self.actionCount[action] += 1
        self.cumulativeReward += reward
        return state, reward, done



    def n_step_TD(self, n=FA_N_STEP, maxIterations=FA_MAX_ITERATIONS):
        state = getState(self.socket, self.stateFormatter, self.debug)
        self.previousState = state
        action = self.chooseAction(state)
        
        reward = None # placeholder that should never be accessed (will throw if it is) 
        # self.rewards.append(reward)

        episodeComplete = False
        ranFor = maxIterations
        T = float('inf')
        for t in range(maxIterations):
            if (self.stateFormatter.breakDueToConstraints()):
                self.log.info("Breaking due to constraints.")
                break

            if (t < T):
                nextState, reward, done = self.stepEnvironment(action)
                nextAction = self.chooseAction(nextState)

                self.states.append(state)
                self.actions.append(action)
                self.rewards.append(reward)

                # if (reward > 0):
                self.log.debug("S: {}".format(["{0:0.2f}".format(i) for i in state]))
                self.log.debug("A: {}".format(action))
                self.log.debug("R: {}".format(reward))
                self.log.debug("S': {}".format(["{0:0.2f}".format(i) for i in nextState]))

                # track state, reward pairs for updates
                # if (len(self.rewards) > 2*n): #queue
                #     self.rewards.pop(0)
                #     self.states.pop(0)
                #     self.actions.pop(0)

                if (done): # Set terminal state
                    self.log.info("Found terminal state! {}".format(nextState))
                    self.numTerminalStates += 1
                    episodeComplete = True
                    ranFor = t
                    self.rewards.append(0)
                    self.states.append(nextState)
                    T = t + 1

            tau = t - n + 1 # tau is the timestep that's being updated
            if (tau >= 0): # there's an update we can do
                update_s, update_a = (self.states[tau], self.actions[tau]) # if (not episodeComplete) else (self.states.pop(0), self.actions.pop(0))
                # We're just updating the weights of the decision we made n steps ago
                idx = min(tau+n, T)

                # for i in range(tau, idx):

                G = np.sum([self.gamma**(i - tau) * self.rewards[i] for i in range(tau,idx)]) 
                # embed()

                # G = np.sum([self.gamma**(idx - i - 1) * self.rewards[i] for i in range(idx)]) 
                self.log.trace("Summed G {}".format(G))
                if (tau+n < T):
                    # with np.errstate(all='raise'):
                    #     try:
                    print(" %d %d " % (tau+n, len(self.states)))
                    idx = (tau+n)
                    G += np.multiply(self.gamma**n, self.value(nextState, self.weightsForActions[nextAction]))
                    self.log.trace("nonterminal G {}".format(G))
                        # except FloatingPointError:
                            # print('oh no!')
                

                self.log.trace("S: {} - A: {}".format(["{0:0.2f}".format(i) for i in update_s], update_a))
                w = self.weightsForActions[update_a]
                A = self.alpha * (G - self.value(update_s, w))
                B = self.gradient(update_s, w)

                self.log.trace("alpha difference {}".format(A))
                self.log.trace("gradient {}".format(["{0:0.3f}".format(i) for i in B]))

                previousWeights = w
                w = w + np.multiply(A, B)

                self.weightsForActions[update_a] = w

                self.log.debug("updated weights {}".format(["{0:0.2f}".format(i) for i in w]))


            if (episodeComplete):
                # do the rest of the learning
                pass
            else:
                action = nextAction
                self.previousState = state
                state = nextState

            if (tau == (T-1)):
                break
            elif (episodeComplete):
                pass
                # print("tau %d T %d" % (tau, T))

            if (self.manual):
                modelPredictions = dict()

                for actionString in ACTION_SET:
                    modelPredictions[actionString] = self.value(state, self.weightsForActions[actionString])

                chosenAction = max(modelPredictions.keys(), key=(lambda k: modelPredictions[k]))
                
                for i, (k,v) in enumerate(modelPredictions.items()):
                    self.log.debug("{}: {:3.2f}".format(k, v))
                self.log.debug("Highest Value Action: {}".format(chosenAction))

        return ranFor
    
if __name__ == "__main__":
    manual = FA_MANUAL

    import sys
    if len(sys.argv) > 1:
        if sys.argv[1] == "-m":
            manual = True

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # sock.connect((HOST, PORT))
    fa  = FA(sock, MANUAL=manual, DEBUG=FA_DEBUG)

    if (FA_READ_WEIGHTS):
        with open(FA_WEIGHTS_PATH) as f:
            data = json.load(f)

        print("Reading in weights ", data)

        for key in data:
            fa.weightsForActions[key] = np.array(data[key])

    cumulativeRewards = []
    episodeLengths = []
    startTime = time.time()
    for j in range(NUM_EPISODES):
        print("Running episode: ", j)
        # sendMessage(sock, "reset")

        time.sleep(0.1)

        fa.reset()
        epLength = fa.n_step_TD()

        episodeLengths.append(epLength)
        cumulativeRewards.append(fa.cumulativeReward)

        print("terminals achieved: ", fa.numTerminalStates)


    runTime = (time.time() - startTime) / 60.
    print("Ran for {:2.2f} minutes.".format(runTime))

    # note that output.json must already exist at this point
    with open(FA_WEIGHTS_PATH, 'w') as f:
        reformatted = dict()
        for key in fa.weightsForActions:
            reformatted[key] = fa.weightsForActions[key].tolist()

        # this would place the entire output on one line
        # use json.dump(lista_items, f, indent=4) to "pretty-print" with four spaces per indent
        json.dump(reformatted, f)



    plt.scatter([i for i in range(len(cumulativeRewards))], cumulativeRewards)
    plt.title("Award Per Episode")
    plt.show()

    plt.scatter([i for i in range(len(episodeLengths))], episodeLengths)
    plt.title("Episode Length")
    plt.show()

    plt.bar(fa.actionCount.keys(), fa.actionCount.values(), 0.5, color='g')
    plt.show()

    