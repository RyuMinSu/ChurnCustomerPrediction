import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import learning_curve
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import classification_report

def plotCols(df, nrows, ncols, target=False, objcols=None): #plotCols함수변경
    if target == True:
        for idx, col in enumerate(objcols):
            plt.subplot(nrows, ncols, idx+1)
            sns.histplot(df, x=col, y="churn")
            plt.title(f"{col}")
        plt.tight_layout()
        plt.show()
    else:
        for idx, col in enumerate(objcols):
            plt.subplot(nrows, ncols, idx+1)
            sns.histplot(df, x=col)
            plt.title(f"{col}")
        plt.tight_layout()
        plt.show()

def Pprob(model, x_train, x_test, y_train, y_test):
    model.fit(x_train, y_train)
    trainPprob = model.predict_proba(x_train)[:,1]
    trainProb = [1 if p>0.5 else 0 for p in trainPprob]
    testPprob = model.predict_proba(x_test)[:,1]
    testProb = [1 if p>0.5 else 0 for p in testPprob]
    return trainPprob, trainProb, testPprob, testProb

def MLperform(model, x_train, x_test, y_train, y_test):
    trainPprob, trainProb, testPprob, testProb = Pprob(model, x_train, x_test, y_train, y_test)
    train_report = classification_report(y_train, trainProb)
    test_report = classification_report(y_test, testProb)        
    trainRoc = roc_auc_score(y_train, trainPprob)
    testRoc = roc_auc_score(y_test, testPprob)
    print(f"\n\t<{model.__class__.__name__}>")
    print(f"train report: roc({trainRoc:.3f})\n {train_report}")
    print(f"test report: roc({testRoc:.3f})\n {test_report}")
    return trainPprob, testPprob, trainRoc, testRoc


def plotLC(model, x_train, y_train, trainsize, cv):
    fig=plt.figure(figsize=(15, 15))
    trainSizes, trainScore, testScore = learning_curve(model, x_train, y_train, train_sizes=trainsize, cv=cv)
    trainMean = np.mean(trainScore, axis=1)
    testMean = np.mean(testScore, axis=1)
    plt.plot(trainSizes, trainMean, "-o", label="train score")
    plt.plot(trainSizes, testMean, "-o", label=f"cross validation score(cv:{cv})")
    plt.title(f"{model.__class__.__name__} Learning Curve", size=20)
    plt.xlabel("Train Sizes", fontsize=15)
    plt.ylabel("Score", fontsize=15)
    plt.legend()
    plt.show()

def plotRocCurve(model, x_train, x_test, y_train, y_test, trainsize, cv):
    trainPprob, testPprob, trainRoc, testRoc = MLperform(model, x_train, x_test, y_train, y_test)
    fprs, tprs, threshold = roc_curve(y_test, testPprob)
    fig = plt.figure(figsize=(15, 15))
    plt.plot(fprs, tprs, label=f"{model.__class__.__name__} roc score: {testRoc:.3f}")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title(f"ROC Curve {model.__class__.__name__}", fontsize=20)
    # plt.axis([-0.01, 1, 0, 1])
    plt.xlabel('False Positive Rate', fontsize=15)
    plt.ylabel('True Positive Rate', fontsize=15)
    plotLC(model, x_train, y_train, trainsize, cv)