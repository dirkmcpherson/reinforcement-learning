import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from IPython import embed
import random
from Gridworld import Gridworld
from Agent import Agent

# Actions is global (bad design)
Actions = {
    0: [0,0],
    1: [1,0],
    2: [-1,0],
    4: [0,1],
    5: [0,-1]
}

def hashStateAction(self, pos, actionKey):
    dx,dy = Actions[actionKey]
    return "x{}y{}dx{}dy{}".format(pos[0],pos[1],dx,dy)

history = dict()
path = set() # list of hashes visit
def updateVisitHistory():
    for key in history:
        if (key in path):
            history[key] = 0
        else:
            history[key] = history[key] + 1

def plotBonusReward():
    steps = 100000
    k = [0.05, 0.01, 0.001] 
    tau = [i for i in range(steps)]
    r = [[(val*np.sqrt(t)) for t in tau] for val in k]

    # plt.figure()
    plt.xlabel('t')
    plt.ylabel('bonus reward')
    plt.title('Bonus Reward over time for unvisited state-actions.')
    plt.plot(tau, r[0], tau, r[1], tau, r[2])
    plt.legend(['k = {}'.format(val) for val in k])
    plt.show()

def takeActionFn(gridworld, actionKey):
    (dx, dy) = Actions[actionKey]
    gridworld.agent.updateHistory(actionKey) # log the action state before actually taking it
    gridworld.moveAgentBy(dx, dy)

def basicQPolicy(agent, Q):
    s0 = agent.position
    a0 = agent.history[-1][1] # action part of history
    hashStateAction(s0, a0)

    ## TODO: If agent reward is not zero it just got something, thats the base of the udpate

    for state, action in agent.history:
        blah = None
        

    # if (a0 in Q):
        # Update value
    # else:
        # Set value to 

def basicQLearning():
    agent = Agent(Actions)
    world = Gridworld(agent, takeActionFn, True)

    numActions = len(agent.Actions)
    Q = dict()

    print("Is this happening? NOPE")
    # Initialize episode
    for i in range(100):
        # Take action
        action = basicQPolicy(agent, Q)

        if (world.GoalState()):
            agent.Reward(100)

        # update Q table
        

        


if __name__ == '__main__':
    # plotBonusReward()

    basicQLearning()