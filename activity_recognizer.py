import DIPPID
from pyqtgraph.flowchart import Flowchart, Node
from pyqtgraph.flowchart.library.common import CtrlNode
import pyqtgraph.flowchart.library as fclib
from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg
from DIPPID_pyqtnode import BufferNode, DIPPIDNode
import numpy as np
from scipy.fft import fft
import sys

class GestureRecognitionNode(Node):
    """
    Node for recording and recognizing gestures
    Contains the SVM logic and UI (ähnlich wie bei der DIPPID node in DIPPID_pyqtnode.py)
    outputs recognized activity if recognizer mode is one
    """
    nodeName = "Gesture Recognition"
    def __init__(self, name):
        terminals = {
            'dataIn': dict(io='in'),
            'dataOut': dict(io='out'),
        }
        Node.__init__(self, name, terminals=terminals)
        # dict for gestures contains name as keys and their traing data as values
        self.gestures = {}
        self.mode = "recognizing"
        self.initUI()

    def initUI(self):
        """
        UI ähnlich wie bei DIPPID node
        Textbox für namen von gesture und add button (eventuell noch delete dazu)
        dann dropdown in der geaddete gestures ausgewählt werden können und start/stop record button, save model button
        theoretisch auch noch
        """
        print("setup UI")

    def train_model(self):
        print("train SVM")

    def safe_model(self):
        # eventuell stattdessen einfach trainingsdaten speichern
        print("save svm")

    def set_mode(self, mode):
        self.mode = mode

    def add_gesture(self, name):
        print("add", name)

    def start_recorder(self):
        self.set_mode("recording")
        print("started recognizer")

    def stop_recorder(self):
        # eventuell noch idel mode hinzufügen
        self.set_mode("recognizing")
        self.train_model()
        print("started recognizer")

    def recognize_gesture(self):
        print("recognized gesture")
        return

    def process(self, **kwds):
        return recognize_gesture

fclib.registerNodeType(GestureRecognitionNode, [('ML', )])

class FFTNode(Node):
    """
    takes acceloremter data as input and outputs their fourier transform
    """
    nodeName = "FFT"

    def __init__(self, name):
        terminals = {
            'inputAccelX': dict(io='in'),
            'inputAccelY': dict(io='in'),
            'inputAccelZ': dict(io='in'),
            'dataOut': dict(io='out'),
        }
        Node.__init__(self, name, terminals=terminals)

    def transform(self, data):
        ffts = []
        for x in data:
            ffts.append(fft(x))
        print (ffts)
        return ffts

    def process(self, **kwds):
        return self.transform([kwds["inputAccelX"], kwds["inputAccelY"], kwds["inputAccelZ"]])

fclib.registerNodeType(FFTNode, [('Data', )])


if __name__ == '__main__':

    app = QtGui.QApplication([])
    win = QtGui.QMainWindow()
    win.setWindowTitle('Analyze DIPPID Data')
    cw = QtGui.QWidget()
    win.setCentralWidget(cw)
    layout = QtGui.QGridLayout()
    cw.setLayout(layout)
    fc = Flowchart(terminals={})
    w = fc.widget()
    layout.addWidget(fc.widget(), 0, 0, 2, 1)

    dippidNode = fc.createNode("DIPPID", pos=(0, 0))
    bufferNodeX = fc.createNode("Buffer", pos=(150, -100))
    bufferNodeY = fc.createNode("Buffer", pos=(150, 0))
    bufferNodeZ = fc.createNode("Buffer", pos=(150, 100))
    fftNode = fc.createNode("FFT", pos=(150, 100))

    fc.connectTerminals(dippidNode['accelX'], bufferNodeX['dataIn'])
    fc.connectTerminals(dippidNode['accelY'], bufferNodeY['dataIn'])
    fc.connectTerminals(dippidNode['accelZ'], bufferNodeZ['dataIn'])
    fc.connectTerminals(bufferNodeX['dataOut'], fftNode['inputAccelX'])
    fc.connectTerminals(bufferNodeY['dataOut'], fftNode['inputAccelY'])
    fc.connectTerminals(bufferNodeZ['dataOut'], fftNode['inputAccelZ'])


    win.show()
    sys.exit(app.exec_())
