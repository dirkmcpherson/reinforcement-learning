from Agent import Agent
from tkinter import *

WALL = 1
GOAL = 2

class Gridworld:
    def __init__(self, agent, learner):
        self.learner = learner # so ugly

        self.display = False
        self.canvas = None
        self.pw = 25
        self.ph = 25

        self.agent = agent
        self.agentIcon = None
        
        self.grid = None

        self.colorMap = {0: 'white', 1: 'black', 2: 'green'}

        self.root = Tk()
        # self.root.withdraw() # dont show the window (prevents tiny window popup in multiple trials)

        # self.world = [
        #     [0,0,0,0],
        #     [0,0,0,0],
        #     [0,0,0,0],
        #     [0,0,0,GOAL]
        # ]

        # self.alternateWorld = [
        #     [0,0,0,0],
        #     [0,0,0,0],
        #     [0,1,1,1],
        #     [0,0,0,GOAL]
        # ]

        # self.world = [
        #     [0,1,0],
        #     [0,1,1],
        #     [0,0,GOAL]
        # ]

        # self.alternateWorld = [
        #     [0,0,0],
        #     [0,0,0],
        #     [1,1,GOAL]
        # ]
        # self.world = [
        #     [0,0,1,1,1,1,1,1,1],
        #     [0,0,0,0,0,0,0,0,0],
        #     [0,1,1,1,1,1,1,1,0],
        #     [0,0,0,0,0,0,0,0,0],
        #     [0,1,1,1,1,0,0,1,0],
        #     [0,1,1,1,1,0,0,1,1],
        #     [0,0,0,0,1,0,0,0,GOAL]
        # ]

        # self.alternateWorld = [
        #     [0,0,1,1,1,1,1,1,1],
        #     [0,0,0,0,0,0,0,0,0],
        #     [1,1,1,1,0,1,1,1,0],
        #     [0,0,0,0,0,0,0,1,0],
        #     [0,1,1,0,1,1,1,1,0],
        #     [0,1,1,0,1,0,0,1,1],
        #     [0,0,0,0,0,0,0,0,GOAL]
        # ]
        # self.world = [
        #     [0,0,0,0,0,0,0,0,0],
        #     [0,0,0,0,0,0,0,0,0],
        #     [1,1,1,1,1,1,1,1,0],
        #     [0,0,0,0,0,0,0,0,0],
        #     [0,0,0,0,0,0,0,0,0],
        #     [0,0,0,0,0,0,0,0,0],
        #     [GOAL,0,0,0,0,0,0,0,0]
        # ]
        # self.alternateWorld = [
        #     [0,0,0,0,0,0,0,0,0],
        #     [0,0,0,0,0,0,0,0,0],
        #     [1,0,1,1,1,1,1,1,0],
        #     [0,0,0,0,0,0,0,0,0],
        #     [0,0,0,0,0,0,0,0,0],
        #     [0,0,0,0,0,0,0,0,0],
        #     [GOAL,0,0,0,0,0,0,0,0]
        # ]
        self.world = [
            [0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0],
            [0,1,1,1,1,1,1,1,1],
            [0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,GOAL]
        ]
        self.alternateWorld = [
            [0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0],
            [1,1,1,1,1,1,1,1,0],
            [0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,GOAL]
        ]
        self.StartPosition = (0,3)

        # TODO: UPDATE THESE WHEN MAP CHANGES
        self.Width = len(self.world)
        self.Height = len(self.world[0])  
    
    def close(self, event):
        print("close")
        self.root.destroy()

    def swapWorlds(self):
        self.world = self.alternateWorld # world with different paths throught it
        self.learner.lookingForNextWin = True

    def acceptInput(self, event):
        # print("Accepting {}".format(event.char))
        c = event.char
        dx = 0
        dy = 0
        if (c == 'w'):
            dy = -1 # hack due to map being mirrored about x-axis
        elif (c == 'a'):
            dx = -1
        elif (c == 's'):
            dy = 1 # hack due to map being mirrored about x-axis
        elif (c == 'd'):
            dx = 1
        else:
            return

        # Figure out which action this corresponds to
        actionKey = None
        for (key, action) in self.agent.Actions.items():
            if (dx == action[0] and dy == action[1]):
                actionKey = key

        if not actionKey:
            print("ERROR: input did not correspond to action.")

        self.learner.manualActionSelection(self, actionKey)

    def bindKeys(self):
        self.root.bind('<Escape>', self.close)
        self.root.bind('<Key>', self.acceptInput)

    def show(self):
        self.display = True
        self.grid = Canvas(self.root, width=self.pw*self.Width, height=self.pw*self.Height)

        self.bindKeys()
   
        for i in range(self.Width):
            for j in range(self.Height):
                color = self.colorMap[self.world[i][j]]
                self.grid.create_rectangle(i * self.pw, j * self.pw, (i+1) * self.pw, (j+1) * self.pw, fill = color, width=1)
        
        self.moveAgentTo(self.StartPosition)
        self.grid.pack(side=LEFT)
        self.grid.focus_set()
        self.root.mainloop()

    def moveAgentBy(self, dx, dy):
        self.moveAgentTo( (self.agent.position[0] + dx, self.agent.position[1] + dy) )

    def moveAgentTo(self, pos):
        (x,y) = pos
        noChange = False
        if (x < 0 or y < 0 or x >= self.Width or y >= self.Height):
            # print("ERROR: invalid agent position {} in world of size {}".format((x,y), (self.Width, self.Height)))
            noChange = True
        elif (self.world[x][y] == WALL):
            noChange = True
        elif (self.world[x][y] == GOAL):
            # print("Yay! Goal! {}".format(self.agent.history))
            self.agent.Reward(self.learner.goalreward)

        if (noChange):
            return

        self.agent.position = [x,y]
        if (self.agentIcon):
            self.grid.delete(self.agentIcon)

        x_offset = self.pw * 0.75
        y_offset = self.ph * 0.75

        if (self.display):
            self.agentIcon = self.grid.create_oval(x * self.pw + x_offset, y * self.pw + y_offset, (x+1) * self.pw - x_offset, (y+1) * self.pw - y_offset, fill = 'yellow')

    def IsGoalState(self,x,y):
        return self.world[x][y] == GOAL

    def GoalState(self):
        return self.world[self.agent.position[0], self.agent.position[1]] == GOAL

    def Reset(self):
        self.moveAgentTo(self.StartPosition)

    def PrintWorld(self):
        for row in self.world:
            print(row)



if __name__ == '__main__':
    Actions = {
        0: [1,0],
        1: [-1,0],
        2: [0,1],
        3: [0,-1]
    }
    a = Agent(Actions)
    gw = Gridworld(a, True)
    gw.show()

