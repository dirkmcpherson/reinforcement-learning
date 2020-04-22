TRACE = 0
DEBUG = 1
INFO = 2
WARN = 3
ERROR = 4

class Logger(object):
    def __init__(self, loglevel = 0):
        self.loglevel = loglevel

    def warn(self,msg):
        self.print(msg, WARN)

    def error(self, msg):
        self.print(msg, ERROR)

    def info(self, msg):
        self.print(msg, INFO)

    def debug(self, msg):
        self.print(msg, DEBUG)

    def trace(self, msg):
        self.print(msg, TRACE)

    def print(self, msg, level):
        if self.loglevel <= level:
            print(msg)