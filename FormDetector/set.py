from smoothnessTracker import SmoothnessTracker
from pathTracer import PathTracer
from repCounter import RepCounter
from repetition import Repetition as Rep
from normaliser import Normaliser
import cv2

class Set(object):
    def __init__(self, cap, classifier):
        self.frame = None
        self.classifier = classifier
        self.width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.initialPositions = []
        self.repetitions = []
        self.smoothnessTracker = SmoothnessTracker(0, cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.normaliser = Normaliser()
        self.repCounter = RepCounter()
        self.repNumber = -1
        self.boundariesSet = False
        self.boundaries = []
        self.barbellCoordinates = []
        self.perfectSet = True
        PathTracer.goodPositions = []
        PathTracer.badPositions = []

    def processNewCoordinate(self, frame, coordinates, currentFrame):
        self.frame = frame
        bbCoord, fwCoord = coordinates
        self.setBoundaries(bbCoord, fwCoord) if not self.boundariesSet else self.drawBoundaries()
        self.displayCurrentRep(bbCoord[1])
        self.appendPositionToRep(coordinates)
        unsmoothRep = self.trackSmoothness(bbCoord[1], currentFrame)
        if unsmoothRep: self.repetitions[-1].setFormBad()

    def setBoundaries(self, bbCoord, fwCoord):
        bottomBoundary, topBoundary = self.repCounter.setBoundaries(bbCoord[1])
        if bottomBoundary and topBoundary is not 0:
            self.initialPositions = self.findCentre(bbCoord), self.findCentre(fwCoord)
            self.boundaries = bottomBoundary, topBoundary
            self.boundariesSet = True
            self.drawBoundaries()

    def findCentre(self, newbox):
        centreX = int(newbox[0]) + int(newbox[2] / 2)
        centreY = int(newbox[1]) + int(newbox[3] / 2)
        return centreX, centreY

    def drawBoundaries(self):
        cv2.line(self.frame, (0, int(0.9 * self.boundaries[0])),
                 (int(self.width), int(0.9 * self.boundaries[0])), (0, 255, 255), 2, 1)
        cv2.line(self.frame, (0, int(1.3 * self.boundaries[1])),
                 (int(self.width), int(1.3 * self.boundaries[1])), (0, 255, 0), 2, 1)
        cv2.line(self.frame, self.initialPositions[0], self.initialPositions[1], (255, 0, 0), 4)

    def displayCurrentRep(self, bbYCoord):
        self.checkRepCount(bbYCoord)
        cv2.putText(self.frame, 'Good Rep Count: ' + str(PathTracer.goodRepCount)
                    + "     Bad Rep Count: "  + str(PathTracer.badRepCount),
                    (int(0.01 * self.width),
                     int(0.1 * self.height)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    def checkRepCount(self, bbYCoord):
        currentRepNum = self.repCounter.incramentRepCount(bbYCoord)
        if currentRepNum is not self.repNumber:
            if self.repNumber > -1: self.checkCurrentRep()
            self.createNewRep()
            self.repNumber = currentRepNum

    def checkCurrentRep(self):
        currentRep = self.repetitions[-1]
        normalisedRep = self.normaliser.normaliseCurretRep(currentRep)
        goodRep = self.classifier.analyseRepetition(normalisedRep)
        print(goodRep)
        if not goodRep[0]:
            currentRep.setFormBad()

    def createNewRep(self):
        if self.repetitions:
            currentRep = self.repetitions[-1]
            currentRep.appendGoodOrBad()
        rep = Rep()
        self.repetitions.append(rep)

    def appendPositionToRep(self, coordinates):
        currentRep = self.repetitions[-1]
        bBcoordinate = self.findCentre(coordinates[0])
        footCoord = self.findCentre(coordinates[1])
        currentRep.appendPosition(self. frame, bBcoordinate,footCoord)

    def trackSmoothness(self, yCoord, frameNum):
        return self.smoothnessTracker.trackSmoothness(yCoord, frameNum, self.repNumber)

    def createPlot(self):
        # self.normaliser.normaliseSet(self.repetitions)
        self.smoothnessTracker.createPlot()



