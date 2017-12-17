import numpy

class CatchAgent:

    def __init__(self):
        self.actions = {
            0: (0, -1),
            1: (0, 1),
            2: (-1, 0),
            3: (1, 0)
        }

        self.position = (0, 0)
        self.positionOld = None

    def move(self, action):
        self.positionOld = self.position
        d = self.delta(action)
        self.position = (self.position[0] + d[0], self.position[1] + d[1])

    def move_random(self):
        action = numpy.random.randint(0, len(self.actions))
        return self.move(action)

    def delta(self, action):
        return self.actions[action]
