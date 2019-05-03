import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def _getDictInNumpy(_dict):


    tt = list(_dict.keys())
    depths = np.array(tt)

    ll = []
    for d in depths:
        ll.append(_dict[d])

    retVal = np.array(ll)

    return depths, retVal


def drawAccuraciesLoc(_accD):

    depths, accNum = _getDictInNumpy(_accD)
    print ("e2157 accNum.shape", accNum.shape)

    fig, axA = createPlotMatrix(1, 2)
    setAxisLabels  (axA, 0, 0, 'depth', 'accuracy')
    setGraphingData(axA, 0, 0, depths, accNum[:, 0], graphColor="tab:blue", label='1st graph')
    setGraphingData(axA, 0, 0, depths, accNum[:, 1], graphColor="tab:green",label='2nd graph')
    setLegends     (axA[0][0], "upper left")


    setAxisLabels  (axA, 0, 1, 'depth', 'accuracy')
    setGraphingData(axA, 0, 1, depths, accNum[:, 2], graphColor="tab:red",  label='3rd graph')
    setLegends     (axA[0][1], "upper left")

    plt.tight_layout()
    plt.show()
    return

def createPlotMatrix(m, n):

    fig = plt.figure()
    axId = 0
    axA = []
    for raw in range(m):
        rawL = []
        axA.append(rawL)
        for col in range(n):
            axId += 1
            rawL.append(fig.add_subplot(m,n,axId))
            # TODO should we add here some additional data?
            pass
        pass

    return fig, axA

def setAxisLabels(_axA, rowId, colId, xLabel, yLabel):
    ax = _axA[rowId][colId]
    ax.set(xlabel=xLabel, ylabel=yLabel)

    return

def setGraphingData(_axA, rowId, colId, x_data, y_data, graphColor="tab:blue", label=''):

    ax = _axA[rowId][colId]
    ax.plot(x_data, y_data, graphColor, label=label)

    return

def setLegends(_ax, _position):
    _ax.legend(loc=_position)
    return

if  __name__ == '__main__':

    accD = {}
    for i in range(1,16):
        accD[i] = np.random.rand(3)

    drawAccuraciesLoc(accD)

    pass