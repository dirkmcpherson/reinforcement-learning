import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from IPython import embed
import random
from Gridworld import Gridworld
from Agent import Agent
from time import time
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
        self.world = Gridworld(self.agent, self, self.manualActionSelection)
        self.Q = np.zeros((self.world.Width, self.world.Height, len(Actions))) # initialize q table to zeros

        # self.world.show()

        self.rewards = []
        self.cumulativeReward = 0

        self.model = self.BuildModel()
    
    def Step(self):
        # self.basicQPolicy(self.world, self.Q)
        self.DynaQPolicy(self.world, self.Q)

    def SelectModelStateAction(self):
        #random for placeholder
        s = [random.randint(0, self.world.Width - 1), random.randint(0, self.world.Height - 1)]
        a = random.randint(0, len(Actions) - 1)
        return s, a

    # The model is structurally the same as the Q table, but Model[S,A] -> S' rather than a value
    def BuildModel(self):
        model = np.zeros((self.world.Width, self.world.Height, len(Actions), 2)) # Every S,A pair should give rise to an S', we could add R here, but instead we'll use the real Q table (that _should_ be the same thing)
        
        # now initialize the model so that every transition leads to S (which we'll update as we actually explore)
        # doin it the loooong way
        w = self.world.Width
        h = self.world.Height
        A = len(Actions)
        for x in range(w):
            for y in range(h):
                for a in range(A):
                    model[x][y][a] = [x,y] # S' starts out as S
        
        return model.astype(int)

    # Our model can be deterministic, which greatly simplifies things
    def ModelStep(self):
        # For Dyna-Q, we simulate a randomly previously observed state and action and update their reward in the Q table
        s, a = self.SelectModelStateAction()

        s_prime = self.model[s[0]][s[1]][a]
        self.UpdateQ(s, a, s_prime, 0)


    def UpdateQ(self, s, a, s_prime, reward):
        # if (np.max(self.Q[s[0]][s[1]][a]) >= 99.9):
        #     embed()

        x,y = s
        xNew, yNew = s_prime
        # Dont do an update if you're entering the terminal state (special case handled by policy)
        if (self.world.IsGoalState(xNew,yNew)):
            return

        alpha = 1.0
        gamma = 0.5
        currentExpectedValue = self.Q[x][y][a]
        newExpectedValue = np.max(self.Q[xNew][yNew][:])
        error = newExpectedValue - currentExpectedValue
        self.Q[x][y][a] = currentExpectedValue + alpha * (reward + (gamma * error))

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

        print("--------------------")
    
    def RestartEpisode(self):
        self.world.Reset()
        self.agent.startNewEpisode()

    def manualActionSelection(self, gridworld, actionKey):
        self.basicQPolicy(gridworld, self.Q, actionKey)

    def DynaQPolicy(self, gridworld, Q, manualActionKey = None):
        endTime = time() + 0.01 # update at 100hz   

        agent = gridworld.agent
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

        # update our model of S,A,S'
        self.model[x][y][actionKey] = [xNew, yNew]

        # Update your previous state with the new reward TD(0)
        reward = agent.GetAndResetReward()

        # Agent only gets direct rewared on episode completion
        # Hack to recognize end of episodes by reward
        if (reward > 0):
            print ("End of episode!")
            self.cumulativeReward += reward
            self.Q[x][y][actionKey] = reward # action key here actually doesn't matter (and really shouldnt be included) but since we do a max over the next state, filling out the action values for the terminal state "shouldnt" have side-effects
            self.RestartEpisode()
            self.PrintQ()
        else:
            self.UpdateQ([x,y], actionKey, [xNew, yNew], reward)

        
        self.rewards.append(self.cumulativeReward)

        modelSteps = 0
        while (time() - endTime < 0):
            self.ModelStep()
            modelSteps = modelSteps + 1
        
        print("ModelSteps ", modelSteps)

        self.PrintQ()   


    def basicQPolicy(self, gridworld, Q, manualActionKey = None):
        agent = gridworld.agent
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
            self.UpdateQ([x,y], actionKey, [xNew, yNew], reward)


        
        self.rewards.append(self.cumulativeReward)
        self.PrintQ()   


if __name__ == '__main__':
    # plotBonusReward()

    # basicQLearning()
    learner = QLearner()

    for i in range(100):
        learner.Step()

    # plt.xlabel('Time Steps')
    # plt.ylabel('Cumulative Reward')
    # x = [i for i in range(len(learner.rewards))]
    # y = learner.rewards
    # # plt.title('Bonus Reward over time for unvisited state-actions.')
    # plt.plot(x,y)
    # plt.show()


