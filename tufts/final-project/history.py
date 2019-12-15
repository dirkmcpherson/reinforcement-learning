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
if __name__ == "__main__":
    pass    