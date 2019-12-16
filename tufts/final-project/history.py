import numpy as np
import copy

class History:
    def __init__(self):
        self.historyLength = 100
        self.record = []

    def push(self, xp):
        if (len(self.record) > self.historyLength):
            # print("history full")
            self.record.pop(0)

        self.record.append(xp)

    def getHistoryInWindow(self, startIdx, windowSize):
        if (startIdx > len(self.record)):
            print("ERROR: history requested outside of range.")
            return []
            
        return self.record[startIdx:(startIdx+windowSize)]

    def relevantHistory(self):
        # Give the consistency of the last n steps (+n for number of steps clockwise, -n for number of steps counterclockise)
        numSteps = 2
        if len(self.record) < numSteps:
            return [0 for i in range(numSteps)]

        relevantHistory = []
        idx = -1
        for i in range(numSteps):
            idx -= i 
            relevantHistory.append(self.record[idx][1])

        # print("relevantHistory: ", relevantHistory)

        return relevantHistory


if __name__ == "__main__":
    pass    