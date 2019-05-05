import matplotlib
from sklearn import tree, model_selection
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report

matplotlib.use('TkAgg')
import pandaUtils
import ModelTuning
import PipeLine_PCA_DecisionTree
from   GraphDisplay import *


def buildModel(max_depth=None, random_state=1):
    model = tree.DecisionTreeClassifier(random_state=random_state, max_depth=max_depth)
    return model

from typing import List, Dict

def prec_scorer(model, ml_data, ml_target):
    return precision_score(ml_target, model.predict(ml_data), average="micro")


def fitAndTestModel(X_test:np.ndarray, X_train, y_test, y_train):

    max_treeDepth = X_test.shape[1]
    accD = {}
    classMetricsD = {}
    reportD = {}
    for depth in range(1, max_treeDepth+1):

        model = buildModel(depth)
        model.fit(X_train, y_train)

        accD[depth]  = [accuracy_score(y_test,   model.predict(X_test)) ,   \
                        accuracy_score(y_train,  model.predict(X_train)),   \
                        accuracy_score(labels,   model.predict(features))]

        tPr = precision_score(y_test,  model.predict(X_test), average="micro")
        tRe = recall_score   (y_test,  model.predict(X_test), average="micro")

        classMetricsD[depth] = [precision_score(y_test,  model.predict(X_test), average="micro"),\
                                recall_score   (y_test,  model.predict(X_test), average="micro")]

        reportD[depth] = classification_report(y_test, model.predict(X_test))
    return accD, classMetricsD, reportD

def printAccuracies(_accD : Dict[int, List[float]]):

    def printSliceOfTestScores(_accD: Dict[int, List[float]], _sliceId: int):
        for depth in _accD.keys():
            print("e1112 depth=%d\ttest.accuracy  = %6.4f" % (depth, _accD[depth][_sliceId]))
        print ("\n\n")
        return

    print ("e1115 accuracies of test set")
    printSliceOfTestScores(_accD, 0)
    print ("e1116 accuracies of train set")
    printSliceOfTestScores(_accD, 1)
    print ("e1117 accuracies of test+train set")
    printSliceOfTestScores(_accD, 2)

    return

def printPrecisions(_accD : Dict[int, List[float]]):

    def printSliceOfTestScores(_accD: Dict[int, List[float]], _sliceId: int):
        for depth in _accD.keys():
            print("e1112 depth=%d\ttest.accuracy  = %6.4f" % (depth, _accD[depth][_sliceId]))
        print ("\n\n")
        return

    print ("e1115 accuracies of test set")
    printSliceOfTestScores(_accD, 0)
    print ("e1116 accuracies of train set")
    printSliceOfTestScores(_accD, 1)
    print ("e1117 accuracies of test+train set")
    printSliceOfTestScores(_accD, 2)

    return


def getDictInNumpy(_dict : Dict[int, List[float]]):


    tt = list(_dict.keys())
    depths = np.array(tt)

    ll = []
    for d in depths:
        ll.append(_dict[d])

    retVal = np.array(ll)

    return depths, retVal

def drawAccuracies(_accD):

    depths, accNum = getDictInNumpy(_accD)
    print ("e2157 accNum.shape", accNum.shape)

    fig, axA = createPlotMatrix(1, 2)
    setAxisLabels  (axA, 0, 0, 'depth', 'accuracy')
    setGraphingData(axA, 0, 0, depths, accNum[:, 0], graphColor="tab:blue", label='accuracy_score for test set')
    setGraphingData(axA, 0, 0, depths, accNum[:, 1], graphColor="tab:green",label='accuracy_score for train set')
    setLegends     (axA[0][0], "upper left")


    setAxisLabels  (axA, 0, 1, 'depth', 'accuracy')
    setGraphingData(axA, 0, 1, depths, accNum[:, 2], graphColor="tab:red",  label='accuracy_score for traibn+test set')
    setLegends     (axA[0][1], "upper left")

    plt.tight_layout()
    plt.show()
    return

def drawPrecisionsAndRecalls(_metrD : Dict[int, List[float]]):

    depths, accNum = getDictInNumpy(_metrD)
    plt.plot(depths, accNum[:,0])
    plt.plot(depths, accNum[:,1])

    #plt.plot(depths, accNum[:,1])

#    plt.plot(depths, accNum[:,0], depths, accNum[:,1], depths, accNum[:,2], depths, accNum[:,3])
    plt.xlabel('depth')
    plt.ylabel('precision_score')
    plt.show()
    return

def printClassificationReports(_repD:Dict[int,str]):
    for depth in _repD.keys():
        print ("e1739 depth=%2d \n" % depth)
        print (_repD[depth])


fileName = "/Volumes/DATA_1TB/Safary_Downloads/ml_data.csv"
features, labels = pandaUtils.readAndPrepareData(fileName, ['Unnamed: 0','ID'],['mpr', 'nux'], 'outputs_class')

# breaking data into train ant test sets using random sampling
X_test, X_train, y_test, y_train  = model_selection.train_test_split(features, labels, test_size=0.80, random_state=314)

accD,metr, repD = fitAndTestModel(X_test, X_train, y_test, y_train)
printAccuracies(accD)
printClassificationReports(repD)

param_grid ={   'PCA__n_components'         : [2,3, 4,5,6,7, 8,9,10,11,12,14, 15], \
                'tree_model__max_depth'     : [1, 3, 5, 7, 10, 14, 15], \
                'tree_model__random_state'  : [0, 314]}

if  PipeLine_PCA_DecisionTree.check_param_grid(param_grid):
    pl = PipeLine_PCA_DecisionTree.getPipeLine()
    ModelTuning.tunePipeLine(pl, param_grid, X_train, y_train)
else:
    print ('e1106 Check your param_grid.')

drawAccuracies(accD)
#drawPrecisionsAndRecalls(metr)
pass
