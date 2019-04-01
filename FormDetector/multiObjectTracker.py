from __future__ import print_function
import sys
import cv2
import numpy as np
from set import Set
from imageRotater import ImageRotater as IR
import time

class MultiTracker(object):
    def __init__(self, gymObjects, videoPath, classifier):
        self.gymObjects = gymObjects
        self.cap = cv2.VideoCapture(videoPath)
        self.classifier = classifier
        self.barbellPositions = []
        self.footWearPositions = []
        self.repetitions = []
        self.rotater = IR(videoPath)

    def checkVideoOrientation(self):
        print("Rotation: " + self.rotater.detectRotation())

    def readVideoFrames(self):
        success, frame = self.cap.read()
        frame = self.rotater.rotateFrame(frame)
        if not success:
            print('Failed to read video')
            sys.exit(1)
        return frame

    def selectBoundingBox(self):
        boundingBoxes = []
        for key, value in self.gymObjects.items():
            if key is 'Gym_Plate':
                success, frame =self.cap.read()
                frame = self.rotater.rotateFrame(frame)
                value['Location'] = self.getAccurateLock(frame, value['Location'],
                                                         self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            boundingBoxes.append(value['Location'])
        # while True:
        #     boundingBox = cv2.selectROI('MultiTracker', frame)
        #     boundingBoxes.append(boundingBox)
            k = cv2.waitKey(0) & 0xFF
            if (k == 113):  # q is pressed
                break
        return boundingBoxes

    def createCSRTTracker(self):
        frame = self.readVideoFrames()
        boundingBoxes = self.selectBoundingBox()
        multiTracker = cv2.MultiTracker_create()
        for boundingBox in boundingBoxes:
            multiTracker.add(cv2.TrackerCSRT_create(), frame, boundingBox)
        return multiTracker

    def displayAndTrack(self):
        self.checkVideoOrientation()
        set = Set(self.cap, self.classifier)
        multiTracker = self.createCSRTTracker()
        frameNo, frameNum = 0, 0
        # fgbg = cv2.createBackgroundSubtractorMOG2()
        # startTime = time.time()
        while self.cap.isOpened():
            success, frame = self.cap.read()
            if not success:
                break
            frame = self.rotater.rotateFrame(frame)

            # get updated location of objects in subsequent frames
            success, boxes = multiTracker.update(frame)
            set.processNewCoordinate(frame, boxes, frameNum)
            #
            # set = Set(frame, newbox)
            # draw tracked objects
            for i, newbox in enumerate(boxes):
                if i==0:
                    bbCentre = self.findCentre(newbox)
                    self.barbellPositions.append(bbCentre)
                    p1 = (int(newbox[0]), int(newbox[1]))
                    p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
                    cv2.rectangle(frame, p1, p2, (0,255,0), 2, 1)
                else:
                    p1 = (int(newbox[0]), int(newbox[1]))
                    p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
                    cv2.rectangle(frame, p1, p2, (0,255,0), 2, 1)
                    self.footWearPositions.append(self.findCentre(newbox))

            # self.findBarCentre(frame, boxes[0], self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            # self.matchBBCentre(frame, boxes[0], width)
            # self.subtractBackground(frame, boxes[0], fgbg, width)
            # self.threshold(frame, boxes[0], bbCentre, width)

            frameNum +=1
            # if frameNum % 100 is 0:
            #     self.driftChecker(frame,boxes[0])
            #     input("Paused")

            height  = frame.shape[0]
            if height>1000:
                ratio = 1000/height
                frame = cv2.resize(frame, (0,0), fx=ratio, fy=ratio)
            width = frame.shape[1]
            if width > 1800:
                ratio = 1800/width
                frame = cv2.resize(frame, (0, 0), fx=ratio, fy=ratio)

            # show frame
            cv2.imshow('MultiTracker', frame)


            # quit on ESC button
            if cv2.waitKey(1) & 0xFF == 27:  # Esc pressed
                break
        # self.measureFrameRate(startTime)
        set.createPlot()
        return (self.barbellPositions, self.footWearPositions)
    #
    # def measureFrameRate(self, startTime):
    #     frames = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
    #     endTime = time.time()
    #     timeTaken = endTime - startTime
    #     print(frames/(timeTaken))

    def driftChecker(self, frame, box):
        width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        print(box)
        lockedbox = self.getAccurateLock(frame, box, width)
        p1 = (int(lockedbox[0]), int(lockedbox[1]))
        p2 = (int(lockedbox[0] + lockedbox[2]), int(lockedbox[1] + lockedbox[3]))
        areaOfOverlap = self.compareOverlap(frame, box, lockedbox)
        areaOfUnion = self.areaOfUnion(box, lockedbox, areaOfOverlap)
        IoU = areaOfOverlap/areaOfUnion
        print(IoU)
        cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
        cv2.imshow('threshold', frame)

    def compareOverlap(self,frame, a, b):
        x = max(a[0], b[0])
        y = max(a[1], b[1])
        w = min(a[0] + a[2], b[0] + b[2]) - x
        h = min(a[1] + a[3], b[1] + b[3]) - y
        p1 = (int(x), int(y))
        p2 = (int(x + w),int(y + h))
        cv2.rectangle(frame, p1, p2, (0, 0, 255), 2, 1)
        if w < 0 or h < 0: return ()  # or (0,0,0,0) ?
        return (w*h)

    def areaOfUnion(self, box, lockedBox, areaOfOverlap):
        area1 = box[2]*box[3]
        area2 = lockedBox[2]*lockedBox[3]
        areaOfUnion = (area1+area2) - areaOfOverlap
        return areaOfUnion

    def findCentre(self, newbox):
        centreX = int(newbox[0]) + int(newbox[2] / 2)
        centreY = int(newbox[1]) + int(newbox[3] / 2)
        return centreX, centreY

    def getAccurateLock(self, frame, box, width):
        bbCentre = self.findCentre(box)
        bbWidth = int(box[2])
        bbHeight = int(box[3])
        y1 = bbCentre[1] - bbHeight if (bbCentre[1] - bbHeight) > 0 else 0
        x1 = bbCentre[0] - bbWidth if (bbCentre[0] - bbWidth) > 0 else 0

        cropped = frame[y1:bbCentre[1] + bbHeight, x1:bbCentre[0] + bbWidth]
        imgray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        circles = cv2.HoughCircles(imgray, cv2.HOUGH_GRADIENT, 1.75, width, 30, 500, int(0.2*bbHeight), int(0.3*bbHeight))
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (x, y, r) in circles:
                cv2.circle(cropped, (x, y), r, (0, 255, 0), 4)
                if (bbCentre[0] - bbWidth) > 0:
                    x = x + (bbCentre[0] - bbWidth)
                if (bbCentre[1] - bbHeight) > 0:
                    y = y + (bbCentre[1] - bbHeight)
                cv2.circle(frame, (x, y), r, (0, 255, 0), 4)
            return x-r, y-r, 2*r, 2*r

    def threshold(self, frame, box, centre, width):
        imgray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        centreThresh = imgray[centre[1], centre[0]]
        print(str(centre) + ':' + str(centreThresh))
        ret, threshold = cv2.threshold(imgray, self.findAverage(centreThresh)-5,
                                       self.findAverage(centreThresh+5), cv2.THRESH_BINARY_INV)
        cv2.imshow('threshold', threshold)
        self.findBarCentre(threshold, box, width)

    def findAverage(self, list):
        total = 0
        for number in list:
            total += number
        return int(total/len(list))


    def findBarCentre(self, frame, box, width):
        bbCentre = self.findCentre(box)
        bbWidth = int(box[2])
        bbHeight = int(box[3])
        cropped = frame[bbCentre[1] - int(0.3 * bbHeight):bbCentre[1] + int(0.3 * bbHeight),
                  bbCentre[0] - int(0.3 * bbWidth):bbCentre[0] + int(0.3 * bbWidth)]
        cv2.imshow('Cropped', cropped)
        imgray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        imgray = cv2.GaussianBlur(imgray, (7, 7), 0)
        edged = cv2.Canny(imgray, 100, 30)
        # cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cv2.imshow('MultiTrackerCnts', imgray)

        circles = cv2.HoughCircles(imgray, cv2.HOUGH_GRADIENT, 1.5, width, None, 100, 30,
                                   int(0.05 * bbHeight), int(0.15 * bbHeight))
        if circles is not None:
            print(circles)
            circles = np.round(circles[0, :]).astype("int")
            for (x, y, r) in circles:
                croppedCircle = imgray[y-int(1.5*r):y+int(1.5*r), x-int(1.5*r):x+int(1.5*r)]
                cv2.circle(imgray, (x, y), r, (0, 255, 0), 4)
                x = x + (bbCentre[0] - int(0.3 * bbWidth))
                y = y + (bbCentre[1] - int(0.3 * bbHeight))
                cv2.circle(frame, (x, y), r, (0, 255, 0), 4)
        cv2.imshow('MultiTrackerGray', imgray)

    def matchBBCentre(self, frame, croppedCircle):
        for pic in listdir('BBCentres'):
            template = cv2.imread('BBCentres/' + pic, 0)
            w, h = template.shape[::-1]
            cW, cH = croppedCircle.shape[::-1]
            if cW <= w or cH <= h:
                return False
            res = cv2.matchTemplate(croppedCircle, template, cv2.TM_CCOEFF_NORMED)
            print(res)
            threshold = 0.6
            loc = np.where(res >= threshold)
            for pt in zip(*loc[::-1]):
                cv2.rectangle(frame, pt, (pt[0] + w, pt[1] + h), (0, 255, 255), 2)
                return True
            return False


    def subtractBackground(self, frame, box, fgbg, width):
        bbCentre = self.findCentre(box)
        bbWidth = int(box[2])
        bbHeight = int(box[3])
        cropped = frame[bbCentre[1] - int(0.3 * bbHeight):bbCentre[1] + int(0.3 * bbHeight),
                  bbCentre[0] - int(0.3 * bbWidth):bbCentre[0] + int(0.3 * bbWidth)]
        fgmask = fgbg.apply(cropped)
        # cv2.imshow('frame', cropped)
        # cv2.imshow('fgmask', fgmask)
        self.findBarCentre(frame, box, fgmask, width)