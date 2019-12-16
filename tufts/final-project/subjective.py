import random
import numpy as np
import time
import learner
from IPython import embed
from feedback import Feedback, Valence
from action import ActionSet

class Subjective:
    def __init__(self, learner):
        # Whether to delay feedback or respond immediately
        self.delay = False
        self.delay_mean = 2.5
        self.delay_stdev = 1.5

        self.feedback = []

        self.learner = learner

        self.windowSize = 20

        self.totalCircles = 0

        self.allmotion = []

    
    def update(self, environment):
        """Respond to the state of the environment
        
        Returns:
            [type] -- [description]
        """

        # no feedback to give
        feedback = self.evaluateEnvironment(environment)
        if (feedback):
            self.queueFeedback(feedback)
        else: #do nothing
            pass

        # if we're waiting to give feedback, check the delay
        if (self.feedback and self.feedback[0].timeToIssue < time.time()):
            self.learner.acceptFeedback(self.feedback.pop(0)) # expensive to use pop but we don't expect enough feedback for it to be a problem

    def queueFeedback(self, feedback):
        timeToIssue = 0
        if (self.delay):
            timeToIssue = time.time() + max(0, np.random.normal(self.delay_mean, self.delay_stdev))
        
        f = Feedback(feedback, timeToIssue)
        print("Should give feedback {} in {} at {}".format(f.count, f.timeToIssue - time.time(), time.time()))
        self.feedback.append(f)

    # def containsCircle(self, thetas):


    def evaluateEnvironment(self):
        history = self.learner.history
        record = history.record
        idxs = [i for i in range(len(record))]
        bestWindow = None
        bestScore = 0
        for i in idxs:
            window = history.getHistoryInWindow(i,self.windowSize)
            best_idx_s = 0
            best_idx_f = 0
            if len(window) < (self.windowSize-1):
                break

            secondSegment = [ActionSet[entry[1]][1]*0.1 for entry in window] # multiplied by environment.dt
            # movementRange = sum(secondSegment)

            # score is based on how many moves were in the same direction
            score = 0
            action = secondSegment[0]
            for idx, entry in enumerate(secondSegment[1:]):
                if (entry != 0) and np.sign(entry) == np.sign(action):
                    if (score == 0):
                        best_idx_s = idx
                    score += 1
                elif (bestScore < score):
                    best_idx_f = idx
                    bestScore = score
                    bestWindow = window[best_idx_s:best_idx_f+1]

                action = entry

        # if (len(self.allmotion) % 1000 == 0):
        #     print("secondSegment: ", [round(entry,2) for entry in secondSegment])
        #     print("score: ", score)
        #     print("================== ", len(self.allmotion))
        self.allmotion.append(bestScore)
        # print("bestwindow: ", bestWindow)
        if (bestWindow is not None):
            # print("Peak score: ", bestScore)
            if bestScore <= 3:
                self.learner.acceptFeedback(bestWindow, -1.)
            else:
                self.learner.acceptFeedback(bestWindow, bestScore / self.windowSize)


            # if (abs(movementRange) >= np.pi/10.):
            #     self.allmotion.append(abs(movementRange))
            #     print("circular motion.")
            #     circle = True
            #     self.learner.acceptFeedback(window, abs(movementRange) / (2*np.pi))
            # else:
            #     pass
                # print("Not enough: ", movementRange)

            # elif (abs(movementRange) < np.pi / 10.):
                # pass
                # print("Not enough movement.")
                # self.learner.acceptFeedback(window, -0.1)

                # embed()
                # time.sleep(1)
                # print("mr ", movementRange)
        # if (circle):
        #     self.learner.acceptFeedback(window)
        #     print("Made a circle!")
        #     self.totalCircles += 1

        


        # return Feedback(fb)


if __name__ == '__main__':
    startTime = time.time()
    o = Subjective(Learner())
    for i in range(100):
        o.update(0) # 0 holds the place of the unimplemented environment class
        time.sleep(0.1)
        print("----{}----".format(time.time() - startTime))