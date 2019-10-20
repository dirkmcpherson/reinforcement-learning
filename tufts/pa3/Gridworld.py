from Agent import Agent
from tkinter import *

WALL = 1
GOAL = 2

class Gridworld:
    def __init__(self, agent, actionFn, show = False):
        self.canvas = None
        self.pw = 100
        self.ph = 100

        self.actionFn = actionFn

        self.agent = agent
        self.agentIcon = None
        
        self.grid = None

        self.colorMap = {0: 'white', 1: 'black', 2: 'green'}

        self.root = Tk()

        self.world = [
            [0,1,0],
            [0,1,1],
            [0,0,GOAL]
        ]
        self.StartPosition = (0,0)

        # TODO: UPDATE THESE WHEN MAP CHANGES
        self.Width = len(self.world)
        self.Height = len(self.world[0])

        if (show):
            self.show()
        
    def close(self, event):
        print("close")
        self.root.destroy()

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

        self.actionFn(self, actionKey)

    def bindKeys(self):
        self.root.bind('<Escape>', self.close)
        self.root.bind('<Key>', self.acceptInput)

    def show(self):
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
            print("ERROR: invalid agent position {} in world of size {}".format((x,y), (self.Width, self.Height)))
            noChange = True
        elif (self.world[x][y] == WALL):
            noChange = True
        elif (self.world[x][y] == GOAL):
            print("Yay! Goal! {}".format(self.agent.history))

        if (noChange):
            return

        self.agent.position = [x,y]
        if (self.agentIcon):
            self.grid.delete(self.agentIcon)

        x_offset = self.pw * 0.75
        y_offset = self.ph * 0.75
        self.agentIcon = self.grid.create_oval(x * self.pw + x_offset, y * self.pw + y_offset, (x+1) * self.pw - x_offset, (y+1) * self.pw - y_offset, fill = 'yellow')

    def GoalState(self):
        return self.world[self.agent.position[0], self.agent.position[1]] == GOAL

    def Reset(self):
        self.moveAgentTo(self.StartPosition)



if __name__ == '__main__':
    a = Agent()
    gw = Gridworld(a, True)
