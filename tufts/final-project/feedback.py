from enum import Enum

class Valence(Enum):
    POSITIVE = 1
    NEGATIVE = 2
    NEUTRAL = 3

class Feedback:
    count = -1
    def __init__(self, valence, timeToIssue=0):
        Feedback.count += 1
        self.count = Feedback.count
        self.timeToIssue = timeToIssue
        self.valence = valence