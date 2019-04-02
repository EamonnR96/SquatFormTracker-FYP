import pandas
from sklearn.tree import DecisionTreeClassifier

class Classifier(object):
    def __init__(self):
        self.trainingFile = "training.csv"
        self.headers = []
        self.dataSet = None
        self.CARTClassifier = None

    def createDecisionTreeClassifier(self):
        self.seperateCSVFile()
        XTrain, yTrain, = self.splitTestAndTrain()
        self.CARTClassifier = DecisionTreeClassifier()
        self.CARTClassifier.fit(XTrain, yTrain)

    def seperateCSVFile(self):
        readCSV = pandas.read_csv(self.trainingFile, header=0)
        readCSV.drop(readCSV.columns[0], axis=1, inplace=True)
        self.headers = list(readCSV.columns.values)
        self.dataSet = readCSV._get_numeric_data()

    def splitTestAndTrain(self):
        X = self.dataSet.drop('FormQuality', axis =1)
        y = self.dataSet['FormQuality']
        return X, y

    def analyseRepetition(self, normalisedRep):
        XTest = self.convertRep(normalisedRep)
        yPred = self.CARTClassifier.predict(XTest)
        return yPred

    def convertRep(self, normalisedRep):
        headers = []
        data = []
        for i in range(50):
            headerX = "X" + str(i)
            headerY = "Y" + str(i)
            headers.append(headerX)
            headers.append(headerY)
            currentPos = normalisedRep[i]
            data.append(currentPos[0])
            data.append(currentPos[1])
        return self.convertToDataFrame(headers, data)

    def convertToDataFrame(self, headers, data):
        df = pandas.DataFrame([data], columns=headers)
        return df
