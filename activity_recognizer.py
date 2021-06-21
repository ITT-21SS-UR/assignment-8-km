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
    nodeName = "GestureRecognizer"

    def __init__(self, name):
        terminals = {
            'dataIn': dict(io='in'),
            'dataOut': dict(io='out'),
        }
        Node.__init__(self, name, terminals=terminals)
        # dict for gestures contains name as keys and their traing data as values
        self.gestures = {}
        self.mode = "recognizing"
        self._init_ui()
        Node.__init__(self, name, terminals=terminals)

    def _init_ui(self):
        print("init UI")
        """
        UI ähnlich wie bei DIPPID node
        Textbox für namen von gesture und add button (eventuell noch delete dazu)
        dann dropdown in der geaddete gestures ausgewählt werden können und start/stop record button, save model button
        theoretisch auch noch
        """
        self.ui = QtGui.QWidget()
        self.layout = QtGui.QGridLayout()

        # UI for adding gestures
        label = QtGui.QLabel("Add new Gesture:")
        self.new_gesture_name = QtGui.QLineEdit()
        self.layout.addWidget(self.new_gesture_name)
        self.layout.addWidget(label)
        self.add_gesture_button = QtGui.QPushButton("add gesture")
        self.add_gesture_button.clicked.connect(lambda: self.add_gesture(self.new_gesture_name.text().strip()))
        self.layout.addWidget(self.add_gesture_button)

        # UI for recording gestures (accumulating training data)
        label2 = QtGui.QLabel("Record a Gesture")
        self.layout.addWidget(label2)

        # drop down menu to choose gesture from list
        self.gesture_to_train = QtGui.QComboBox()
        self.gesture_to_train.addItems(self.gestures.keys())
        self.layout.addWidget(self.gesture_to_train)

        # button to start recording
        self.start_button = QtGui.QPushButton("start recording")
        self.start_button.clicked.connect(self.start_recorder)
        self.layout.addWidget(self.start_button)

        # button to stop recording
        self.stop_button = QtGui.QPushButton("stop recording")
        self.stop_button.clicked.connect(self.stop_recorder)
        self.layout.addWidget(self.stop_button)
        self.stop_button.setEnabled(False)

        self.ui.setLayout(self.layout)
        print("setup UI")

    def ctrlWidget(self):
        return self.ui

    def train_model(self):
        print("train SVM")

    def safe_model(self):
        # eventuell stattdessen einfach trainingsdaten speichern
        print("save svm")

    def set_mode(self, mode):
        self.mode = mode

    def add_gesture(self, name):
        if name not in self.gestures and name != "":
            self.gestures[name] = []
            self.gesture_to_train.clear()
            self.gesture_to_train.addItems(self.gestures.keys())
        else:
            sys.stderr.write("The gesture name either already exists or is empty. Please choose another name")
        print("Gestures:", self.gestures)

    def start_recorder(self):
        self.set_mode("recording")
        self.gesture_to_train.setEnabled(False)
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        print("started recognizer")

    def stop_recorder(self):
        # eventuell noch idel mode hinzufügen
        self.set_mode("idle")
        self.train_model()
        self.gesture_to_train.setEnabled(True)
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        print("started recognizer")

    def recognize_gesture(self):
        print("recognized gesture")
        return

    def process(self, **kwds):
        if self.mode == "recording":
            self.gestures[self.gesture_to_train.currentText()].append(kwds["dataIn"])
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
        print("start")

    def transform(self, data):
        ffts = []
        for x in data:
            ffts.append(fft(x))
        return ffts

    def process(self, **kwds):
        fft_result = self.transform([kwds["inputAccelX"], kwds["inputAccelY"], kwds["inputAccelZ"]])[0]
        return {'dataOut': fft_result}


fclib.registerNodeType(FFTNode, [('Data', )])


if __name__ == '__main__':

    # set up app
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

    # create flowchart nodes
    dippidNode = fc.createNode("DIPPID", pos=(0, 0))
    recognizerNode = fc.createNode("GestureRecognizer", pos=(200, 100))
    bufferNodeX = fc.createNode("Buffer", pos=(150, -100))
    bufferNodeY = fc.createNode("Buffer", pos=(150, 0))
    bufferNodeZ = fc.createNode("Buffer", pos=(150, 100))
    fftNode = fc.createNode("FFT", pos=(150, 100))

    # connect flowchart nodes
    fc.connectTerminals(dippidNode['accelX'], bufferNodeX['dataIn'])
    fc.connectTerminals(dippidNode['accelY'], bufferNodeY['dataIn'])
    fc.connectTerminals(dippidNode['accelZ'], bufferNodeZ['dataIn'])
    fc.connectTerminals(bufferNodeX['dataOut'], fftNode['inputAccelX'])
    fc.connectTerminals(bufferNodeY['dataOut'], fftNode['inputAccelY'])
    fc.connectTerminals(bufferNodeZ['dataOut'], fftNode['inputAccelZ'])
    fc.connectTerminals(fftNode['dataOut'], recognizerNode['dataIn'])

    # start app
    win.show()
    sys.exit(app.exec_())
