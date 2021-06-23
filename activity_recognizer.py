"""
Overall structure for training the classifier taken from "Wiimote - FFT - SVM" notebook
script created by both team members (Distribution: 80% Kay, 20% Marco)

It might take some for the node to switch between to a recognized gesture as the buffernodes will still send data
from previous movement for some time (as they send the last 32 values)
Tested gesture were:
still (phone laying still on the table)
shake (shaking the phone in your hand)
turn (turning the phone around its X Axis)
"""

import DIPPID
from pyqtgraph.flowchart import Flowchart, Node
from pyqtgraph.flowchart.library.common import CtrlNode
import pyqtgraph.flowchart.library as fclib
from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg
from DIPPID_pyqtnode import BufferNode, DIPPIDNode
import numpy as np
from sklearn import svm
from scipy.fft import fft
import sys
import pandas as pd
import json


class GestureRecognitionNode(Node):
    """
    Node for recording and recognizing gestures
    Contains the classifier logic and UI
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
        # sets mode off node, ca either be "recording" when recording gestures or recognizing for classifying gestures
        self.mode = "recognizing"
        self.recognizer = svm.SVC()
        # dict for classifications of gestures contains classifier number as key and the gesture's name as value
        self.classifiers = {}
        self.is_trained = False
        self._init_ui()
        Node.__init__(self, name, terminals=terminals)

    # creates UI for node
    def _init_ui(self):
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

        # text box for gesture output
        self.recognized_gesture_label = QtGui.QLabel("Recognized gesture: ")
        self.layout.addWidget(self.recognized_gesture_label)

        self.ui.setLayout(self.layout)

    def ctrlWidget(self):
        return self.ui

    # trains svm model
    def train_model(self):
        if len(self.gestures.keys()) > 1:
            x = 0
            categories = []
            training_data = []

            # creates categories and training data for training model
            for key in self.gestures.keys():
                category_numbers = [x]*len(self.gestures[key])
                categories.extend(category_numbers)
                training_data.extend(self.gestures[key])
                # set dictionary entry with classifier number as key and gesture name as value
                self.classifiers[str(x)] = key
                x += 1

            # svm classifier needs training data for at least 2 gestures
            try:
                self.recognizer.fit(training_data, categories)
                self.is_trained = True

            except ValueError:
                pass
        else:
            print("needs more gesture categories before training")

    def set_mode(self, mode):
        self.mode = mode

    def add_gesture(self, name):
        if name not in self.gestures and name != "":
            self.gestures[name] = []
            self.gesture_to_train.clear()
            self.gesture_to_train.addItems(self.gestures.keys())
        else:
            sys.stderr.write("The gesture name either already exists or is empty. Please choose another name")

    # start recording of gesture
    def start_recorder(self):
        self.set_mode("recording")
        self.recognized_gesture_label.setText("Recognized gesture: ")
        self.gesture_to_train.setEnabled(False)
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.add_gesture_button.setEnabled(False)

    # stops recording of gesture and calls train model function
    def stop_recorder(self):
        self.train_model()
        self.gesture_to_train.setEnabled(True)
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.add_gesture_button.setEnabled(True)
        self.set_mode("recognizing")

    # classifies gesture return classifier number of recognized gesture
    def recognize_gesture(self, data):
        prediction = self.recognizer.predict([data])
        return prediction

    def process(self, **kwds):
        print(mode)
        if self.mode == "recording":
            self.gestures[self.gesture_to_train.currentText()].append(kwds["dataIn"])
        if self.mode == "recognizing":
            if self.is_trained:
                recognized_classifier = self.classifiers[str(self.recognize_gesture(kwds["dataIn"])[0])]
                self.recognized_gesture_label.setText("Recognized gesture: " + recognized_classifier)
        return


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

    # fourier transforms averages of acceloremeter data
    def transform(self, data):
        j = len(data[0])
        if j != 32:
            return
        i = 0
        avgs = []
        while i < j:
            avgs.append(sum([x[i] for x in data]) / 3)
            i += 1
        transformed = np.abs(fft(avgs)/len(avgs))[1:len(avgs)//2]
        return transformed

    def process(self, **kwds):
        fft_result = self.transform([kwds["inputAccelX"], kwds["inputAccelY"], kwds["inputAccelZ"]])
        return {'dataOut': fft_result}


fclib.registerNodeType(FFTNode, [('Data', )])


if __name__ == '__main__':

    # set up app
    app = QtGui.QApplication([])
    win = QtGui.QMainWindow()
    win.resize(800, 1000)
    win.setWindowTitle('Analyze DIPPID Data')
    cw = QtGui.QWidget()
    win.setCentralWidget(cw)
    layout = QtGui.QGridLayout()
    cw.setLayout(layout)
    fc = Flowchart(terminals={})
    layout.addWidget(fc.widget(), 0, 0, 2, 1)

    # create flowchart nodes
    dippidNode = fc.createNode("DIPPID", pos=(-300, -300))
    recognizerNode = fc.createNode("GestureRecognizer", pos=(150, -300))
    bufferNodeX = fc.createNode("Buffer", pos=(-150, -450))
    bufferNodeY = fc.createNode("Buffer", pos=(-150, -300))
    bufferNodeZ = fc.createNode("Buffer", pos=(-150, -150))
    fftNode = fc.createNode("FFT", pos=(0, -300))

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
