import csv
import os

class Normaliser(object):
    def __init__(self):
        self.maxXDiff = 0
        self.maxYDiff = 0
        self.footCoord = 0

    def normaliseCurretRep(self, rep):
        self.setFootPosInRep(rep)
        self.setLocalMax(rep)
        repCoords = self.normaliseToFoot(rep)
        repCoords = self.reducePointsToFifty(repCoords)
        return repCoords

    def normaliseSet(self, repetitions):
        self.getFootCoord(repetitions)
        self.getMaxValues(repetitions)
        del(repetitions[-1])
        normalisedReps = []
        for rep in repetitions:
            repCoords = self.normaliseToFoot(rep)
            repCoords = self.reducePointsToFifty(repCoords)
            normalisedReps.append(repCoords)
        self.writeToCSVFile(normalisedReps)

    def getFootCoord(self, repetitions):
        i = int(len(repetitions)/2)
        midRep = repetitions[i]
        self.setFootPosInRep(midRep)

    def setFootPosInRep(self, midRep):
        numberOfFootCoords = len(midRep.footCoordinates)
        footCoord = midRep.footCoordinates[int(numberOfFootCoords/2)]
        self.footCoord = footCoord

    def getMaxValues(self, set):
        yVal = []
        xVal = []
        for rep in set:
            for val in rep.listOfBBCoordinates:
                yVal.append(val[1])
                xVal.append(val[0])
        self.maxXDiff = max(xVal) - min(xVal)
        self.maxYDiff = self.footCoord[1] - min(yVal)

    def setLocalMax(self, rep):
        yVal = []
        xVal = []
        for val in rep.listOfBBCoordinates:
            yVal.append(val[1])
            xVal.append(val[0])
        self.maxXDiff = max(xVal) - min(xVal)
        self.maxYDiff = self.footCoord[1] - min(yVal)

    def normaliseToFoot(self, rep):
        normalisedCoords = []
        rep = self.clearDataBeforeGoingDown(rep)
        for coord in rep.listOfBBCoordinates:
            relativeCoord = self.subtractFootFromBar(coord)
            normalisedCoord = self.divideByMaxes(relativeCoord)
            normalisedCoords.append(normalisedCoord)
        return normalisedCoords

    def subtractFootFromBar(self, coord):
        bbX = coord[0]
        bbY = coord[1]
        relativeX = bbX - self.footCoord[0]
        relativeY = self.footCoord[1] - bbY
        return ([relativeX, relativeY])

    def divideByMaxes(self, relativeCoord):
        normX = relativeCoord[0]/self.maxXDiff
        normY = relativeCoord[1]/self.maxYDiff
        return [normX, normY]

    def clearDataBeforeGoingDown(self, rep):
        startPoint = self.detectdownwardMovement(rep)
        rep.listOfBBCoordinates = rep.listOfBBCoordinates[startPoint:]
        return rep

    def detectdownwardMovement(self, rep):
        goingDownCount = 0
        previousY = 0
        for i, position in enumerate(rep.listOfBBCoordinates):
            if position[1] > previousY:
                previousY = position[1]
                goingDownCount += 1
                if goingDownCount > 15: return i-15
            else:
                previousY = position[1]
                goingDownCount = 0

    def reducePointsToFifty(self, repCoords):
        normalisedArray = []
        stepSize = float(len(repCoords)) / float(50)
        step = 0
        if len(repCoords) > 50:
            for i in range(50):
                step += stepSize
                arrayItem = round(step) - 1
                normalisedArray.append(repCoords[arrayItem])
        elif len(repCoords) < 50:
            for i in range(50):
                normalisedArray.append(self.calculateWeightAverage
                                       (repCoords,step))
                step += stepSize
        else:
            normalisedArray = repCoords
        return normalisedArray

    def calculateWeightAverage(self, array, itemNo):
        print(array)
        poisition1 = array[int(itemNo)]
        try:
            position2 = array[int(itemNo) + 1]
        except:
            position2 = poisition1
        remainder = itemNo % 1
        averageX = poisition1[0]*(1-remainder) + position2[0]*remainder
        averageY = poisition1[1]*(1-remainder) + position2[1]*remainder
        position = (averageX, averageY)
        return position

    def writeToCSVFile(self, normalisedSet):
        trainingFile = 'training.csv'
        if os.path.exists(trainingFile):
            self.appendToFile(trainingFile,normalisedSet)
        else:
            self.writeInitialFile(trainingFile, normalisedSet)

    def writeInitialFile(self,trainingFile, normalisedSet):
        with open(trainingFile, mode='w') as trainFile:
            fieldnames = self.generateFieldNames()
            trainWriter = csv.writer(trainFile, delimiter=',',
                                     quotechar='"', quoting=csv.QUOTE_MINIMAL)

            trainWriter.writerow(fieldnames)
            for i, rep in enumerate(normalisedSet):
                self.writeRow(trainWriter, rep, i)

    def appendToFile(self, trainingFile, normalisedSet):
        startRow = self.getRowNum(trainingFile)
        with open(trainingFile, mode='a') as trainFile:
            trainWriter = csv.writer(trainFile, delimiter=',',
                                     quotechar='"', quoting=csv.QUOTE_MINIMAL)

            for i, rep in enumerate(normalisedSet):
                self.writeRow(trainWriter, rep, startRow+i)

    def generateFieldNames(self):
        headers = ['']
        for i in range(50):
            xLabel = 'X' + str(i)
            yLabel = 'Y' +str(i)
            headers.append(xLabel)
            headers.append(yLabel)
        return headers

    def writeRow(self,trainWriter, rep, repNum):
        row = [str(repNum)]
        for position in rep:
            row.append(position[0])
            row.append(position[1])
        trainWriter.writerow(row)

    def getRowNum(self, trainingFile):
        with open(trainingFile, mode='r') as readFile:
            trainReader = csv.reader(readFile, delimiter=',',
                                     quotechar='"', quoting=csv.QUOTE_MINIMAL)
            rowCount = sum(1 for row in trainReader)
            print(rowCount)
        return rowCount - 1











