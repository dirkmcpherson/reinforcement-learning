import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from IPython import embed
import random
import math
from Gridworld import Gridworld
from Agent import Agent
from time import time

# Actions is global (bad design)
Actions = {
    0: [1,0],
    1: [-1,0],
    2: [0,1],
    3: [0,-1]
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

def plotBonusReward(steps):
    k = [0.005, 0.01, 0.001] 
    tau = [i for i in range(steps)]
    r = [[(val*np.sqrt(t)) for t in tau] for val in k]

    # plt.figure()
    plt.xlabel('t')
    plt.ylabel('bonus reward')
    plt.title('Bonus Reward over time for unvisited state-actions.')
    plt.plot(tau, r[0], tau, r[1], tau, r[2])
    plt.legend(['k = {}'.format(val) for val in k])
    plt.show()

class QLearner():
    def __init__(self, dyna=False, plus=False, experiment=False):
        self.randomizeAction = 0.1
        self.agent = Agent(Actions)
        self.world = Gridworld(self.agent, self)
        self.Q = np.zeros((self.world.Width, self.world.Height, len(Actions))) # initialize q table to zeros

        self.goalreward = 1
        self.rewards = []
        self.cumulativeReward = 0
        self.completedEpisodes = 0
        self.stepsPerEpisode = []

        self.updatePolicy = self.basicQPolicy
        self.PLUS = plus
        self.EXPERIMENT = experiment
        if (dyna):
            self.updatePolicy = self.DynaQPolicy

        # For Dyna-Q
        self.numModelUpdates = 50
        self.model = self.BuildModel()

        # For Dyna-Q+, a table of how long its been since a state-action was visited, and an incrementer for easy addition
        # if (self.PLUS):
        #     self.randomizeAction = 1.0
        self.timestep = 0
        self.history = dict() #self.BuildHistory()

        self.lookingForNextWin = False
        self.timeSinceLooking = 0

        self.visitCount = np.zeros((self.world.Width, self.world.Height))

        # standardized random number generator for action selection
        self.random = random.Random()
        self.random.seed(12)
    
    def Step(self):
        self.timestep = self.timestep + 1
        # if (self.PLUS and (self.randomizeAction > 0.1)):
        #     self.randomizeAction -= 0.001
        self.updatePolicy(self.world, self.Q)

    def SelectModelStateActionDynaQPlus(self, fromState = None):
        s = random.choice(list(self.history.keys()))
        a = self.random.randint(0, len(Actions) - 1)

        # Hack to promote exploration (over favoring states that have tiny q-values because they've been visited before)
        # if (0 in self.Q[s[0]][s[1]][:]):
        #     print("Choosing unexplored option")
        #     a = self.random.choice( np.argwhere(self.Q[s[0]][s[1]] == 0) )[0]
        
        r = self.RecencyBonus(s[0], s[1], a)
        if math.isnan(r):
            r = 0
        return s, a, r

    def SelectModelStateActionDynaQ(self):
        #random for placeholder
        s = random.choice(list(self.history.keys()))
        actionCounts = self.history[s]
        a = self.random.choice( np.argwhere(actionCounts != 0) )[0]
        # a = self.random.randint(0, len(Actions) - 1)
        return s, a

    # The model is structurally the same as the Q table, but Model[S,A] -> S' rather than a value
    def BuildModel(self):
        model = np.zeros((self.world.Width, self.world.Height, len(Actions), 3)) # Every S,A pair should give rise to an S', we could add R here, but instead we'll use the real Q table (that _should_ be the same thing)
        
        # now initialize the model so that every transition leads to S (which we'll update as we actually explore)
        # doin it the slooooow, but clear way
        w = self.world.Width
        h = self.world.Height
        A = len(Actions)
        r = 0
        for x in range(w):
            for y in range(h):
                for a in range(A):
                    model[x][y][a] = [x,y,r] # S' starts out as S
        
        return model.astype(int)

    def BuildHistory(self):
        return -1 * np.ones((self.world.Width, self.world.Height, len(Actions), 1))

    # Our model can be deterministic, which greatly simplifies things
    def ModelStep(self):
        # For Dyna-Q, we simulate a randomly previously observed state and action and update their reward in the Q table
        r = 0 
        if (self.PLUS) and not (self.EXPERIMENT): #and (self.timestep > 1000):
            s, a, r = self.SelectModelStateActionDynaQPlus()
            # print("plus update reward ", r)
        else:
            s, a = self.SelectModelStateActionDynaQ()
            
        [x, y ,r_transition] = self.model[s[0]][s[1]][a]
        s_prime = (x,y)
        r = r + r_transition

        before = self.Q[s[0]][s[1]][a]
        self.UpdateQ(s, a, s_prime, r)
        after = self.Q[s[0]][s[1]][a]

        # if (self.PLUS):
        #     print("Model updated Q from {:6.2f} to {:6.2f}".format(before, after))

    def RecencyBonus(self,x,y,a):
        k = 0.001

        # if you've never tried the queried state-action (For experiment) give a large number
        timeSince = 0
        if ((x,y) in list(self.history.keys())):
            timeSince = self.history[(x,y)][a]

        # if (self.timestep > 1000):
            # embed()

        dt = self.timestep - timeSince
        return k * np.sqrt(dt)

    def UpdateQ(self, s, a, s_prime, reward, PLUS_MODEL = False):
        x,y = s
        xNew, yNew = s_prime
        # Dont do an update if you're entering the terminal state (special case handled by policy)
        if (self.world.IsGoalState(xNew,yNew)):
            return

        alpha = 0.1
        gamma = 0.95

        newExpectedValue = gamma * np.max(self.Q[xNew][yNew][:])
        currentExpectedValue = self.Q[x][y][a]
        error = newExpectedValue - currentExpectedValue
        self.Q[x][y][a] = currentExpectedValue + alpha * (reward + error)

    def PrintQ(self, Q = "blarg"):
        # allow for manual entry of a Q table
        if Q == "blarg": 
            Q = self.Q
        # For debugging go through the Q and pix the max state action
        debuggingView = []
        for i in range(len(Q)):
            row = []
            for j in range(len(Q[0])):
                row.append("{:6.2f}".format(np.max(Q[i][j])))
            debuggingView.append(row)

        for entry in debuggingView:
            print(entry)

        print("--------------------")
    
    def RestartEpisode(self):
        self.world.Reset()
        self.agent.startNewEpisode()

    def takeActionFn(self, actionKey):
        # for dyna-q+, update history
        # self.history = self.history + (1 * (self.history != -1)) # only increment valid values (visited values)

        # Update the history
        x,y = self.agent.position
        if ((x,y) in self.history.keys()):
            self.history[(x,y)][actionKey] = self.timestep
        else:
            actionHistory = np.zeros(len(Actions))
            actionHistory[actionKey] = self.timestep
            self.history[(x,y)] = actionHistory

        (dx, dy) = Actions[actionKey]
        self.world.agent.updateHistory(actionKey) # log the action state before actually taking it
        self.world.moveAgentBy(dx, dy)

        # print("{} -> {} by {}".format((x,y), self.agent.position, (dx,dy)))
        
        self.visitCount[self.agent.position[0]][self.agent.position[1]] = self.visitCount[self.agent.position[0]][self.agent.position[1]] + 1

    def manualActionSelection(self, gridworld, actionKey):
        self.basicQPolicy(gridworld, self.Q, actionKey)

    def DynaQPolicy(self, gridworld, Q, manualActionKey = None):
        agent = gridworld.agent
        x,y = agent.position

        actionKey = None
        if (manualActionKey):
            actionKey = manualActionKey
            print("Took action key ",actionKey)
        else:
            # whats the best action we could take from here?
            if (self.random.random() <= self.randomizeAction):
                actionKey = self.random.randint(0, len(Actions) - 1)
            elif (self.EXPERIMENT): #and (self.timestep > 1000)):
                actionVals = Q[x][y][:]
                bestAction = 0
                bestValue = -1
                for i in range(len(actionVals)):
                    v = actionVals[i] + self.RecencyBonus(x,y,i)
                    if (v > bestValue):
                        bestValue = v
                        bestAction = i
                actionKey = bestAction
            else:
                actionKey = np.random.choice(np.argwhere(Q[x][y][:] == np.max(Q[x][y][:])).flatten())
                # actionKey = np.argmax(Q[x][y][:])
                # print("Selected {} val {} from {}".format(actionKey, Q[x][y][actionKey],Q[x][y][:]))

        # Take that action
        self.takeActionFn(actionKey)
        xNew, yNew = agent.position

        # Update your previous state with the new reward TD(0)
        reward = agent.GetAndResetReward()

        # update our model of S,A,S'
        self.model[x][y][actionKey] = [xNew, yNew, reward]

        # Agent only gets direct rewared on episode completion
        # Hack to recognize end of episodes by reward
        if (reward > 0):
            # print ("End of episode!")
            self.stepsPerEpisode.append(len(agent.history))
            self.cumulativeReward += reward
            self.Q[x][y][actionKey] = reward # action key here actually doesn't matter (and really shouldnt be included) but since we do a max over the next state, filling out the action values for the terminal state "shouldnt" have side-effects
            self.RestartEpisode()
            self.completedEpisodes = self.completedEpisodes + 1
            if (self.lookingForNextWin):
                self.lookingForNextWin = False
        else:
            self.UpdateQ([x,y], actionKey, [xNew, yNew], reward)

        if (self.lookingForNextWin):
            self.timeSinceLooking = self.timeSinceLooking + 1

        
        self.rewards.append(self.cumulativeReward)

        for i in range(self.numModelUpdates):
            self.ModelStep()

        
        # print("ModelSteps ", modelSteps)

        # self.PrintQ()   

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
            if (self.random.random() <= 0.1):
                actionKey = self.random.randint(0, len(Actions) - 1)
            else:

                actionKey = np.argmax(Q[x][y][:])

        # Take that action
        self.takeActionFn(actionKey)
        xNew, yNew = agent.position

        # Update your previous state with the new reward TD(0)
        reward = agent.GetAndResetReward()

        # Agent only gets direct reward on episode completion
        # Hack to recognize end of episodes by reward
        if (reward > 0):
            # print ("End of episode!")
            self.stepsPerEpisode.append(len(agent.history))
            self.cumulativeReward += reward
            self.Q[x][y][actionKey] = reward # action key here actually doesn't matter (and really shouldnt be included) but since we do a max over the next state, filling out the action values for the terminal state "shouldnt" have side-effects
            self.RestartEpisode()
            self.completedEpisodes = self.completedEpisodes + 1
        else:
            self.UpdateQ([x,y], actionKey, [xNew, yNew], reward)


        
        self.rewards.append(self.cumulativeReward)
        # self.PrintQ()   


if __name__ == '__main__':
    experimentCount = 5 #100

    iterations = 6000
    # plotBonusReward(iterations)

    
    q_learner_all_rewards = []
    dynaQ_learner_all_rewards = []
    dynaQPlus_learner_all_rewards = []
    dynaQPlus_pa_learner_all_rewards = []

    learners = []
    for j in range(experimentCount):

        # basicQLearning()
        q_learner = QLearner()
        dynaQ_learner = QLearner(True)
        dynaQPlus_learner = QLearner(True, True)
        dynaQPlus_pa_learner = QLearner(True, True, True)

        # learners = [q_learner, dynaQ_learner, dynaQPlus_learner]
        # learners = [dynaQ_learner, dynaQPlus_learner, dynaQPlus_pa_learner]
        # learners = [dynaQ_learner, dynaQPlus_learner]
        learners = [dynaQPlus_learner, dynaQPlus_pa_learner]

        # hack to detect first win post change
        for i in range(iterations):
            [learner.Step() for learner in learners]
            # q_learner.Step()
            # dynaQ_learner.Step()
            # dynaQPlus_learner.Step()

            if (i == int(iterations / 2)):
                print("-----MAP SWAP-----")
                # dynaQPlus_learner.PrintQ((dynaQ_learner.Q - dynaQPlus_learner.Q))
                # dynaQPlus_learner.world.PrintWorld()

                [learner.world.swapWorlds() for learner in learners]
                [learner.RestartEpisode() for learner in learners]
                # q_learner.world.swapWorlds()
                # dynaQ_learner.world.swapWorlds()
                # dynaQPlus_learner.world.swapWorlds()


        q_learner_all_rewards.append(q_learner.rewards)
        dynaQ_learner_all_rewards.append(dynaQ_learner.rewards)
        dynaQPlus_learner_all_rewards.append(dynaQPlus_learner.rewards)
        dynaQPlus_pa_learner_all_rewards.append(dynaQPlus_pa_learner.rewards)


        # dynaQPlus_learner.PrintQ()
        # print("DynaQ")
        # print(dynaQ_learner.visitCount)
        # dynaQ_learner.PrintQ()
        # print("DynaQ+")
        # print(dynaQPlus_learner.visitCount)
        # dynaQPlus_learner.PrintQ()


    # dynaQPlus_learner.PrintQ((dynaQ_learner.Q - dynaQPlus_learner.Q))
    # q_learner.PrintQ()
    # print("--------------")
    # q_learner.world.PrintWorld()

    # print("Completed Episodes ", q_learner.completedEpisodes, dynaQ_learner.completedEpisodes, dynaQPlus_learner.completedEpisodes)
    # print("Steps-per-Episode ", np.mean(q_learner.stepsPerEpisode), np.mean(dynaQ_learner.stepsPerEpisode), np.mean(dynaQPlus_learner.stepsPerEpisode))
    # print("Found goal after swap ", q_learner.timeSinceLooking, dynaQ_learner.timeSinceLooking, dynaQPlus_learner.timeSinceLooking)

    plt.xlabel('Time Steps')
    plt.ylabel('Cumulative Reward')
    x = [i for i in range(iterations)]
    # y0 = q_learner.rewards
    y1 = np.mean(np.array(dynaQ_learner_all_rewards), 0)
    y2 = np.mean(np.array(dynaQPlus_learner_all_rewards), 0)
    y3 = np.mean(np.array(dynaQPlus_pa_learner_all_rewards), 0)
    # y3 = dynaQPlus_pa_learner.rewards

    # embed()
    # plt.title('Bonus Reward over time for unvisited state-actions.')
    # plt.plot(x,y2)
    # plt.legend(["DynaQ+"])
    plt.title("Cumulative value for DynaQ Algorithms")
    plt.xlabel("Timestep")
    plt.ylabel("Cumulative Value")
    # plt.plot(x,y2)
    # plt.plot(x,y1,x,y2)
    plt.plot(x,y2,x,y3)
    # plt.plot(x,y1,x,y2,x,y3)
    # plt.plot(x,y0,x,y1,x,y2)
    # plt.legend(["DynaQ", "DynaQ+"])
    plt.legend(["DynaQ+", "Variant_DynaQ+"])
    plt.show()


