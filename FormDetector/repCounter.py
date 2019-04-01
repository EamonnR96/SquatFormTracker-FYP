

class RepCounter(object):
    def __init__(self):
        self.listOfY = []
        self.repCount = 0
        self.yAxisPreviousPosition = 0
        self.downDirectionCounter = 0
        self.upDirectionCounter = 0
        self.wentDown = False
        self.wentUp = False
        self.bottomBoundary = 0
        self.topBoundary = 0
        self.midpoint = 0
        self.midpointCrossings = 0
        self.wentBelowBoundary = False
        self.wentAboveBoundary = False
        self.smallDownDCounter = 0
        self.smallUpDCounter = 0

    def setBoundaries(self, yCoord):
        self.listOfY.append(yCoord)
        if self.wentDown and self.checkUpwardDirection(yCoord) and self.bottomBoundary is 0:
            self.bottomBoundary = self.listOfY[-10]
        else:
            self.checkInitDownwardDirection(yCoord)
        if self.wentUp and self.checkDownwardDirection(yCoord) and self.topBoundary is 0:
            self.topBoundary = self.listOfY[-10]
        else:
            self.checkInitUpwardDirection(yCoord)
        if self.topBoundary and self.bottomBoundary is not 0:
            self.repCount += 1
            self.midpoint = (self.bottomBoundary + self.topBoundary)/2
        self.yAxisPreviousPosition = yCoord
        return self.bottomBoundary, self.topBoundary


    def checkInitDownwardDirection(self, yCoord):
        if self.downDirectionCounter < 15 and self.checkDownwardDirection(yCoord):
            self.downDirectionCounter +=1
        elif self.downDirectionCounter >= 15 and self.checkDownwardDirection(yCoord):
            self.wentDown = True
        elif self.checkUpwardDirection(yCoord):
            self.downDirectionCounter = 0

    def checkInitUpwardDirection(self, yCoord):
        if self.upDirectionCounter < 15 and self.checkUpwardDirection(yCoord):
            self.upDirectionCounter += 1
        elif self.upDirectionCounter >= 15 and self.checkUpwardDirection(yCoord):
            self.wentUp = True
        elif self.checkDownwardDirection(yCoord):
            self.upDirectionCounter = 0

    def checkDownwardDirection(self, yCoord):
        if (self.yAxisPreviousPosition - yCoord) <= 0 and self.smallDownDCounter > 10:
            self.smallUpDCounter = 0
            return True
        if (self.yAxisPreviousPosition - yCoord) <= 0:
            self.smallDownDCounter += 1
            self.smallUpDCounter = 0
        return False

    def checkUpwardDirection(self, yCoord):
        if (self.yAxisPreviousPosition - yCoord) > 0 and self.smallUpDCounter > 10:
            self.smallDownDCounter = 0
            return True
        if (self.yAxisPreviousPosition - yCoord) > 0:
            self.smallUpDCounter += 1
            self.smallDownDCounter = 0
        return False

    def incramentRepCount(self, yCoord):
        self.checkMidpointCrossed(yCoord)
        if yCoord >= 0.9 * self.bottomBoundary:
            self.wentBelowBoundary = True
        if (yCoord -20) <= 1.3 * self.topBoundary:
            self.wentAboveBoundary = True
        if self.wentAboveBoundary and self.wentBelowBoundary\
                and self.midpointCrossings >= 2 and self.checkGoingDown(yCoord):
            self.repCount += 1
            self.wentBelowBoundary = False
            self.wentAboveBoundary = False
            self.midpointCrossings = 0
        if self.midpointCrossings is 0:
            self.wentBelowBoundary = False
            self.wentAboveBoundary = False
        self.yAxisPreviousPosition = yCoord
        return self.repCount

    def checkGoingDown(self, yCoord):
        if (self.yAxisPreviousPosition - yCoord) < 0:
            return True

    def checkMidpointCrossed(self, yCoord):
        if self.yAxisPreviousPosition <= self.midpoint and yCoord > self.midpoint:
            self.midpointCrossings += 1
        elif self.yAxisPreviousPosition >= self.midpoint and yCoord < self.midpoint:
            self.midpointCrossings += 1




