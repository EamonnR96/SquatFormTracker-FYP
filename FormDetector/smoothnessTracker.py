import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import scipy.signal as sig
import math
import numpy as np

class SmoothnessTracker(object):
    def __init__(self, initY, frameRate, totalFrames):
        self.repetitions = {}
        self.changeInY = []
        self.prevY = initY
        self.frameRate = frameRate
        self.totalFrames = totalFrames
        self.frameToRep = ([], [])
        self.slowingUp = 0
        self.slowingDown = 0
        self.goingUpCount = 0
        self.goingDownCount = 0
        self.previousSlowUpFrame = 0
        self.previousSlowDownFrame = 0
        self.yDiffthisRep = []

    def trackSmoothness(self, yCoord, currentFrame, repNumber):
        self.repetitions.setdefault(repNumber, ([], []))[1].append(currentFrame/self.frameRate)
        self.frameToRep[0].append(currentFrame/self.frameRate)
        self.frameToRep[1].append(repNumber)
        diff = self.compareY(yCoord)
        self.repetitions.setdefault(repNumber, ([], []))[0].append(diff)
        self.prevY = yCoord
        self.changeInY.append(diff)
        unsmoothRep = self.detectBreakInSmoothness(diff, repNumber, currentFrame)
        return unsmoothRep

    def compareY(self, yCoord):
        diff = self.prevY - yCoord
        return diff

    def getMaxDiff(self):
        maxVal = 0
        for i, yValue in enumerate(self.changeInY):
            if i>0:
                if yValue > maxVal:
                    maxVal = yValue
        return maxVal

    def getMinDiff(self):
        minVal = 0
        for i, yValue in enumerate(self.changeInY):
            if i>0:
                if yValue < minVal:
                    minVal = yValue
        return minVal

    def createPlot(self):
        maxDiff = self.getMaxDiff()
        minDiff = self.getMinDiff()
        del(self.changeInY[0])
        del(self.frameToRep[0][0])
        del(self.frameToRep[1][0])
        length = self.totalFrames/self.frameRate
        yhat = sig.savgol_filter(self.changeInY, 51, 5)
        peaks = sig.find_peaks(yhat)
        plt.ylabel('Change In Y Value')
        plt.xlabel('Repetitions')
        # plt.plot(self.frameToRep[0], yhat)
        plt.plot(self.frameToRep[0], self.changeInY, '#ff9900')
        plt.axis([0, int(length), int(minDiff), math.ceil(maxDiff)])
        ticks, labels = self.findWhereRepChanges()
        plt.xticks(ticks, labels)
        plt.show()

    def findWhereRepChanges(self):
        ticks = []
        labels = []
        prevValue = 0
        for i, value in enumerate(self.frameToRep[1]):
            if prevValue is not value:
                ticks.append(self.frameToRep[0][i])
                labels.append(value)
                prevValue = value
        return ticks, labels

    def detectBreakInSmoothness(self, yDiff, currentRep, currentFrame):
        self.yDiffthisRep.append(yDiff)
        self.directionCheck(yDiff)
        maxDiff = max(self.yDiffthisRep)
        minDiff = min(self.yDiffthisRep)
        if currentRep >= 1:
            if self.changeInY[-2] > maxDiff/2 and yDiff <= maxDiff/2 and self.goingUpCount>10:
                if self.slowingUp == 0:
                    self.goingUpCount = 3
                    self.previousSlowUpFrame = currentFrame
                    self.slowingUp += 1
                elif self.slowingUp >= 1 and self.checkFrameDifference(currentFrame):
                    print("Going up Count: " + str(self.goingUpCount))
                    print("Not Smooth Going Up: " + str(currentRep + 1))
                    return True
            if self.changeInY[-2] < minDiff/2 and yDiff >= minDiff/2 and self.goingDownCount>15:
                if self.slowingDown == 0:
                    self.goingDownCount = 5
                    self.previousSlowDownFrame = currentFrame
                    self.slowingDown += 1
                elif self.slowingDown >= 1 and self.checkFrameDifference(currentFrame):
                    print("Not smooth going down: " + str(currentRep + 1))
            if currentRep is not self.frameToRep[1][-2]:
                self.yDiffthisRep = []
                self.slowingDown, self.slowingUp, self.goingDownCount, self.goingUpCount = 0, 0, 0, 0
        return False

    def directionCheck(self, yDiff):
        if yDiff > 0:
            self.goingUpCount += 1
            self.goingDownCount = 0
        if yDiff < 0:
            self.goingDownCount += 1
            self.goingUpCount = 0

    def checkFrameDifference(self, currentFrame):
        if currentFrame - self.previousSlowDownFrame > 5:
            return True
        if currentFrame - self.previousSlowUpFrame > 5:
            return True









