import random
import numpy as np
from state import StateFormatter, generateRandomState


actionSet = [i for i in range(10)]
alpha = 0.1
gamma = 0.5

def terminalState(state):
    return random.random() > 0.98

epsilon = 0.1
def epsilonGreedy(state, weightsForActions):
    if (random.random() < epsilon):
        return random.choice(actionSet)
    else:
        modelPredictions = []
        for action in actionSet:
            modelPredictions.append(model(state, weightsForActions[action]))

        return np.argmax(modelPredictions)
        

def initWeights(state):
    # We want a single weight associated with each entry in the state
    w = np.random.rand(state.shape)
    return w


def model(state, weights):
    return state * weights

def gradient(state, weights):
    # special case for linear model, x(S) is the features of s, for now, since we have no features its just s
    return state

'''
Attempting to do episodic semi-gradient SARSA
'''
def main():
    weightsHistory = []

    # get state from minecraft
    state = generateRandomState()

    # init a set of weights for each action (each has its model)
    weightsForActions = dict()
    for action in actionSet:
        weightsForActions[action] = initWeights(state)



    episodes = 10 
    action = epsilonGreedy(state, weightsForActions)
    for i in range(episodes):
        # get state from minecraft

        # Take action A to transition to the next State and get the Reward (S,A)
        nextState, reward = generateRandomState() # TODO: state given action
        
        w = weightsForActions[action]
        if (terminalState(nextState)):
            w = w + alpha * (reward - model(state, w)) * gradient(state, w)
            continue

        # linear model, so just multiple each member of the state by its weight
        nextAction = epsilonGreedy(nextState, weightsForActions)
        w = w + alpha * (reward + gamma * model(nextState, weightsForActions[nextAction]) - model(state, w)) * gradient(state, w)

        # update weights 
        weightsForActions[action] = w





if __name__ == "__main__":
    main()