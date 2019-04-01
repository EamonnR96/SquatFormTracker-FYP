import numpy as np
import pandas
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from scipy import interp

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import learning_curve


class Classifier(object):
    def __init__(self):
        self.trainingFile = "training.csv"
        self.headers = []
        self.dataSet = None
        self.gaussianClassifier = None
        self.x = None
        self.y= None

    def createSVMClassifier(self):
        self.seperateCSVFile()
        XTrain, yTrain, = self.splitTestAndTrain()
        self.gaussianClassifier= DecisionTreeClassifier()
        self.gaussianClassifier.fit(XTrain, yTrain)
        # self.testClassifier(XTest, yTest)

    def seperateCSVFile(self):
        readCSV = pandas.read_csv(self.trainingFile, header=0)
        readCSV.drop(readCSV.columns[0], axis=1, inplace=True)
        self.headers = list(readCSV.columns.values)
        self.dataSet = readCSV._get_numeric_data()

    def splitTestAndTrain(self):
        X = self.dataSet.drop('FormQuality', axis =1)
        y = self.dataSet['FormQuality']
        self.x = X
        self.y = y
        XTrain, XTest, yTrain, yTest = train_test_split(X, y, test_size=0.1)
        return X, y

    def testClassifier(self):
        XTest = self.x
        yTest = self.y
        yPred = self.gaussianClassifier.predict(XTest)
        print(confusion_matrix(yTest, yPred))
        print(classification_report(yTest, yPred))

    def plotRoC(self):
        X = self.x
        y = self.y
        random_state = np.random.RandomState(0)
        cv = StratifiedKFold(n_splits=6)
        classifier = svm.SVC(kernel='linear', probability=True,
                             random_state=random_state)

        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)

        i = 0
        for train, test in cv.split(X, y):
            probas_ = classifier.fit(X, y).predict_proba(X)
            # Compute ROC curve and area the curve
            fpr, tpr, thresholds = roc_curve(y, probas_[:, 1])
            tprs.append(interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)
            plt.plot(fpr, tpr, lw=1, alpha=0.3,
                     label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

            i += 1
        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
                 label='Chance', alpha=.8)

        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        plt.plot(mean_fpr, mean_tpr, color='b',
                 label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
                 lw=2, alpha=.8)

        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                         label=r'$\pm$ 1 std. dev.')

        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        plt.show()

    def plot_learning_curve(self, estimator=GaussianNB(), ylim=None, cv=None,
                            n_jobs=None, train_sizes=np.linspace(.1, 1.0, 20)):
        X = self.x
        y = self.y
        title = 'Learning Curve(NB)'
        """
        Generate a simple plot of the test and training learning curve.

        Parameters
        ----------
        estimator : object type that implements the "fit" and "predict" methods
            An object of that type which is cloned for each validation.

        title : string
            Title for the chart.

        X : array-like, shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape (n_samples) or (n_samples, n_features), optional
            Target relative to X for classification or regression;
            None for unsupervised learning.

        ylim : tuple, shape (ymin, ymax), optional
            Defines minimum and maximum yvalues plotted.

        cv : int, cross-validation generator or an iterable, optional
            Determines the cross-validation splitting strategy.
            Possible inputs for cv are:
              - None, to use the default 3-fold cross-validation,
              - integer, to specify the number of folds.
              - :term:`CV splitter`,
              - An iterable yielding (train, test) splits as arrays of indices.

            For integer/None inputs, if ``y`` is binary or multiclass,
            :class:`StratifiedKFold` used. If the estimator is not a classifier
            or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

            Refer :ref:`User Guide <cross_validation>` for the various
            cross-validators that can be used here.

        n_jobs : int or None, optional (default=None)
            Number of jobs to run in parallel.
            ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
            ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
            for more details.

        train_sizes : array-like, shape (n_ticks,), dtype float or int
            Relative or absolute numbers of training examples that will be used to
            generate the learning curve. If the dtype is float, it is regarded as a
            fraction of the maximum size of the training set (that is determined
            by the selected validation method), i.e. it has to be within (0, 1].
            Otherwise it is interpreted as absolute sizes of the training sets.
            Note that for classification the number of samples usually have to
            be big enough to contain at least one sample from each class.
            (default: np.linspace(0.1, 1.0, 5))
        """
        plt.figure()
        plt.title(title)
        if ylim is not None:
            plt.ylim(*ylim)
        plt.xlabel("Training examples")
        plt.ylabel("Score")
        train_sizes, train_scores, test_scores = learning_curve(
            estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        plt.grid()

        plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1, color="g")
        plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")

        plt.legend(loc="best")
        return plt

    def analyseRepetition(self, normalisedRep):
        XTest = self.convertRep(normalisedRep)
        yPred = self.gaussianClassifier.predict(XTest)
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

classifier = Classifier()
classifier.createSVMClassifier()
classifier.testClassifier()
# classifier.plotRoC()
classifier.plot_learning_curve()
plt.show()


