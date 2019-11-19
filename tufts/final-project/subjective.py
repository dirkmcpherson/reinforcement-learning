import random
import numpy as np
import time
import learner
from IPython import embed
from feedback import Feedback, Valence

class Subjective:
    def __init__(self, learner):
        # Whether to delay feedback or respond immediately
        self.delay = False
        self.delay_mean = 2.5
        self.delay_stdev = 1.5

        self.feedback = []

        self.learner = learner

    
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

    def evaluateEnvironment(self, environment):
        fb = Valence.NEUTRAL
        # if we like the state of the environment or not
        # r = random.random()
        # if (r > 0.97):
        #     fb = FeedbackType.POSITIVE
        # elif (r > 0.9):
        #     fb = FeedbackType.POSITIVE
        
        if (environment.momentum[0] > 0):
            fb = Valence.POSITIVE
        else:
            fb = Valence.NEGATIVE

        return Feedback(fb)


if __name__ == '__main__':
    startTime = time.time()
    o = Subjective(Learner())
    for i in range(100):
        o.update(0) # 0 holds the place of the unimplemented environment class
        time.sleep(0.1)
        print("----{}----".format(time.time() - startTime))