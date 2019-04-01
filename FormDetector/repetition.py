from pathTracer import PathTracer

class Repetition(object):
    def __init__(self):
        self.listOfBBCoordinates = []
        self.footCoordinates = []
        self.goodForm = True
        self.pathTracer = PathTracer()

    def appendPosition(self, frame, BBcoordinate, footCoordinate):
        self.listOfBBCoordinates.append(BBcoordinate)
        self.footCoordinates.append(footCoordinate)
        self.pathTracer.drawTrace(frame, self.listOfBBCoordinates)

    def appendGoodOrBad(self):
        if self.goodForm:
            for postition in self.listOfBBCoordinates:
                PathTracer.goodPositions.append(postition)
            PathTracer.goodRepCount += 1
        else:
            for postition in self.listOfBBCoordinates:
                PathTracer.badPositions.append(postition)
            PathTracer.badRepCount += 1

    def setFormBad(self):
        self.goodForm = False

