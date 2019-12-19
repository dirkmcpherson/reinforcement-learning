import numpy as np
import copy
from IPython import embed

class History:
    def __init__(self):
        self.historyLength = 100
        self.relevantEntries = 3
        self.record = []

    def push(self, xp):
        if (len(self.record) > self.historyLength):
            # print("history full")
            self.record.pop(0)

        self.record.append(xp)
        # print("history push ", xp)

    def getHistoryInWindow(self, startIdx, windowSize):
        if (startIdx > len(self.record)):
            print("ERROR: history requested outside of range.")
            return []
            
        return self.record[startIdx:(startIdx+windowSize)]

    def relevantHistory(self):
        relevantHistory = []
        if len(self.record) < self.relevantEntries:
            return [0 for i in range(self.relevantEntries)]

        idx = -self.relevantEntries
        for i in range(self.relevantEntries):
            # embed()
            relevantHistory.append(self.record[idx][1])
            idx += 1

        return relevantHistory


if __name__ == "__main__":
    pass    