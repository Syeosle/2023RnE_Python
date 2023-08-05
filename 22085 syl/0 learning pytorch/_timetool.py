import time as t


class Stopwatch:
    def __init__(self):
        self.init = t.time()
        self.last = self.init

    def reset(self):
        self.__init__()

    def term(self):
        now = t.time()
        res = now - self.last
        self.last = now
        return res
