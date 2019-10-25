class Agent:
    def __init__(self, actions):
        self.position = [0,0]
        self.history = [] # (state, action)
        self.Actions = actions
        self.reward = 0

    def updateHistory(self, action):
        self.history.append((self.position, action))

    def startNewEpisode(self):
        self.history = []

    def Reward(self, r):
        self.reward = r

    def GetAndResetReward(self):
        ret = self.reward
        self.reward = 0
        return ret

    