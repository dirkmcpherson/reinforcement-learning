import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from IPython import embed
import random
from Gridworld import Gridworld
from Agent import Agent
import random

# Actions is global (bad design)
Actions = {
    0: [0,0],
    1: [1,0],
    2: [-1,0],
    3: [0,1],
    4: [0,-1]
}

def hashStateAction(pos, actionKey):
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

class QLearner():
    def __init__(self):
        self.agent = Agent(Actions)
        # embed()
        self.world = Gridworld(self.agent, self, self.manualActionSelection)
        self.Q = np.zeros((self.world.Width, self.world.Height, len(Actions))) # initialize q table to zeros
        # self.world.show()

        self.rewards = []
        self.cumulativeReward = 0
    
    def Step(self):
        self.basicQPolicy(self.world, self.Q)

    def PrintQ(self):
        # For debugging go through the Q and pix the max state action
        debuggingView = []
        for i in range(len(self.Q[0])):
            row = []
            for j in range(len(self.Q)):
                row.append(np.max(self.Q[j][i]))
            debuggingView.append(row)

        for entry in debuggingView:
            print(entry)
    
    def RestartEpisode(self):
        self.world.Reset()
        self.agent.startNewEpisode()

    def manualActionSelection(self, gridworld, actionKey):
        self.basicQPolicy(gridworld, self.Q, actionKey)

    def basicQPolicy(self, gridworld, Q, manualActionKey = None):
        agent = gridworld.agent
        s0 = agent.position
        x,y = agent.position
        # a0 = agent.history[-1][1] # action part of history

        actionKey = None
        if (manualActionKey):
            actionKey = manualActionKey
            print("Took action key ",actionKey)
        else:
            # whats the best action we could take from here?
            if (random.random() >= 0.1):
                actionKey = random.randint(0, len(Actions) - 1)
            else:
                actionKey = np.argmax(Q[x][y][:])

        # Take that action
        takeActionFn(gridworld, actionKey)
        xNew, yNew = agent.position

        # Update your previous state with the new reward TD(0)
        alpha = 1.0
        gamma = 0.5
        reward = agent.GetAndResetReward()

        # Agent only gets direct rewared on episode completion
        # Hack to recognize end of episodes by reward
        if (reward > 0):
            print ("End of episode!")
            self.cumulativeReward += reward
            self.Q[x][y][actionKey] = reward # action key here actually doesn't matter (and really shouldnt be included) but since we do a max over the next state, filling out the action values for the terminal state "shouldnt" have side-effects
            self.RestartEpisode()
        else:
            currentExpectedValue = Q[x][y][actionKey]
            newExpectedValue = np.max(Q[xNew][yNew][:])
            error = newExpectedValue - currentExpectedValue
            self.Q[x][y][actionKey] = currentExpectedValue + alpha * (reward + (gamma * error))

        
        self.rewards.append(self.cumulativeReward)
        self.PrintQ()

        print("--------")
        
        # if (reward > 0):
        #     print ("End of episode!")
        #     # self.Q[xNew][yNew][actionKey] = reward # action key here actually doesn't matter (and really shouldnt be included) but since we do a max over the next state, filling out the action values for the terminal state "shouldnt" have side-effects
        #     self.RestartEpisode()

        #     # For debugging go through the Q and pix the max state action
        #     debuggingView = []
        #     for i in range(len(Q[0])):
        #         row = []
        #         for j in range(len(Q)):
        #             row.append(np.max(Q[j][i]))
        #         debuggingView.append(row)

        #     for entry in debuggingView:
        #         print(entry)
        # for entry in reversed(self.Q):
        #     print(entry)

    # sa_key = hashStateAction(s0, a0)

    ## TODO: If agent reward is not zero it just got something, thats the base of the update
    # reward = agent.GetAndResetReward()
    # if (reward > 0):
    #     Q[sa_key] = reward

    # for state, action in agent.history:
    #     blah = None
        

    # if (a0 in Q):
        # Update value
    # else:
        # Set value to 

# def basicQLearning():
#     agent = Agent(Actions)
#     world = Gridworld(agent, takeActionFn, True)

#     numActions = len(agent.Actions)
#     Q = np.zeros(world.Width, world.Height, numActions) # initialize q table to zeros

#     print("Is this happening? NOPE")
#     # Initialize episode
#     for i in range(100):
#         # Take action
#         action = basicQPolicy(agent, Q)

#         # update Q table

#     print
        

        


if __name__ == '__main__':
    # plotBonusReward()

    # basicQLearning()
    learner = QLearner()

    for i in range(10000):
        learner.Step()

    plt.xlabel('Time Steps')
    plt.ylabel('Cumulative Reward')
    x = [i for i in range(len(learner.rewards))]
    y = learner.rewards
    # plt.title('Bonus Reward over time for unvisited state-actions.')
    plt.plot(x,y)
    plt.show()


