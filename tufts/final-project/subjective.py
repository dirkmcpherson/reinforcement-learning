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

        self.windowSize = 50

        self.totalCircles = 0

    
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
        fb = Valence.NEUTRAL
        history = self.learner.history
        record = history.record
        idxs = [i for i in range(len(record))]
        circle = False
        for i in idxs:
            window = history.getHistoryInWindow(i,self.windowSize)
            if len(window) < (self.windowSize-1):
                break

            # secondSegment = [entry[0][1] for entry in window]
            # secondSegment = [ActionSet[entry[1]] for entry in window][1]
            # p0 = secondSegment[0]
            # movementRange = sum([entry - p0 for entry in secondSegment[1:]])
            secondSegment = [ActionSet[entry[1]][1]*0.1 for entry in window]
            movementRange = sum(secondSegment)

            if (abs(movementRange) >= np.pi):
                print("Got circle.")
                circle = True
                self.learner.acceptFeedback(window, abs(movementRange) / (2*np.pi))
            elif (abs(movementRange) < np.pi / 10.):
                print("Not enough movement.")
                self.learner.acceptFeedback(window, -0.1)
            else:
                pass
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