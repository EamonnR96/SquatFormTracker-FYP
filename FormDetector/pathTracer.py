import cv2

class PathTracer(object):
    goodPositions = []
    badPositions = []
    goodRepCount = 0
    badRepCount = 0

    def __init__(self):
        self.positions = []

    def drawTrace(self, frame, positions):
        self.drawPreviousReps(frame)
        for i, coordinate in enumerate(positions):
            if i > 0: self.drawLines(frame, coordinate, positions[i-1], (255, 0, 0))

    def drawPreviousReps(self, frame):
        for i, coordinate in enumerate(PathTracer.goodPositions):
            if i > 0: self.drawLines(frame, coordinate, PathTracer.goodPositions[i - 1], (0, 255, 0))
        for i, coordinate in enumerate(PathTracer.badPositions):
            if i > 0: self.drawLines(frame, coordinate, PathTracer.badPositions[i - 1], (0, 0, 255))

    def drawLines(self, frame, coordinate1, coordinate2, colour):
        cv2.line(frame, coordinate1, coordinate2, colour, 1, 1)
