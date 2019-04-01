from gymObjectDetector import GymObjectDetector
from multiObjectTracker import MultiTracker
from classifier import Classifier

videoList = [
    # "vlc-record-2019-02-16-17h15m39s-VID_20190216_151505.mp4-.mp4"
    # "HalfReps.mp4",
    # "HalfReps-converted.mp4",
    # "PoorForm.mp4",
    # "PoorForm-converted.mp4",
    # "PoorForm2.mp4",
    # "RisingPause.mp4",
    # "RisingPause-converted.mp4",
    # "RisingPause2-converted.mp4",
    # "GoodForm16.mp4",
    # "GoodForm16-converted.mp4",
    # "GoodForm17.mp4"
    # "GoodForm17-converted.mp4",
    # "GoodForm18.mp4"
    # "GoodForm18-converted.mp4",
    # "GoodForm1-converted.mp4",
    # "GoodForm2-converted.mp4",
    # "GoodForm5-converted.mp4",
    # "GoodForm7-converted.mp4",
    # "GoodForm8-converted.mp4",
    # "GoodForm10-converted.mp4",
    # "GoodForm11-converted.mp4"
    # "GoodForm13-converted.mp4",
    # "GoodForm14-converted.mp4",
    # "GoodForm15-converted.mp4",
    # "BB Back Squat (Side Angle)  Volt Athletics-converted.mp4",
    # "Barbell Back Squats (Side View)-converted.mp4"
    #
    # "GoodForm1.mp4",
    # "GoodForm2.mp4",
    # "GoodForm3.mp4",
    # "GoodForm6.mp4",
    # "GoodForm7.mp4",
    # "GoodForm8.mp4"
    # "GoodForm10.mp4",
    # "GoodForm11.mp4",
    # "GoodForm12.mp4"
    # "GoodForm13.mp4",
    # "GoodForm14.mp4",
    # "GoodForm15.mp4",
    # "BB Back Squat (Side Angle)  Volt Athletics.mp4",
    # "Barbell Back Squats (Side View).mp4"
    #
    # "Test1(GoodForm).mp4",
    # "Test2(GoodForm).mp4",
    # "Test11(AllGoodBFrame).mp4",
    # "Test12(6Good).mp4",
    #
    # "Test5(3Good3Struggle).mp4",
    # "Test6(1GoodRestStruggle).mp4",
    # "Test7(HalfReps).mp4",
    # "Test8(PoorForm).mp4",
    # "Test10(3Good3Bad).mp4"
    # "Detector/DLTest.mp4"
    # "Train1(1Good5Bad).mp4",
    # "Train2(1Good6Bad).mp4"
    # "Train3(1Good5Bad).mp4",
    # "Train4(1Good5Bad).mp4",

    ]

for video in videoList:
    videoPath = "/home/eamonn/FYP/Videos/" + video

    gymObjects = {'Gym_Plate': {'Location': '',
                                    'Frame': 0},
                    'FootWear': {'Location': [],
                                    'Frame': 0}
                  # 'Person': {'Location': [],
                  #                   'Frame': 0}
                      }


    classifier = Classifier()
    classifier.createSVMClassifier()
    god = GymObjectDetector(gymObjects, videoPath)
    trackedObjects = god.getNormalisedObjectLocations()
    kcfTracker = MultiTracker(gymObjects, videoPath, classifier)
    barbellPosition, footwearPosition = kcfTracker.displayAndTrack()

