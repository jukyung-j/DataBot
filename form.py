# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'form.ui'
##
## Created by: Qt User Interface Compiler version 6.0.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import *
from PySide6.QtGui import *
from PySide6.QtWidgets import *


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(800, 600)
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.tabWidget = QTabWidget(self.centralwidget)
        self.tabWidget.setObjectName(u"tabWidget")
        self.tabWidget.setGeometry(QRect(0, 0, 791, 561))
        self.Preprocess = QWidget()
        self.Preprocess.setObjectName(u"Preprocess")
        self.groupBox = QGroupBox(self.Preprocess)
        self.groupBox.setObjectName(u"groupBox")
        self.groupBox.setGeometry(QRect(30, 40, 611, 71))
        self.openbtn = QPushButton(self.groupBox)
        self.openbtn.setObjectName(u"openbtn")
        self.openbtn.setGeometry(QRect(10, 20, 75, 24))
        self.open = QTextBrowser(self.groupBox)
        self.open.setObjectName(u"open")
        self.open.setGeometry(QRect(100, 20, 481, 31))
        self.open.setCursorWidth(1)
        self.groupBox_2 = QGroupBox(self.Preprocess)
        self.groupBox_2.setObjectName(u"groupBox_2")
        self.groupBox_2.setGeometry(QRect(30, 130, 341, 131))
        self.gridLayoutWidget = QWidget(self.groupBox_2)
        self.gridLayoutWidget.setObjectName(u"gridLayoutWidget")
        self.gridLayoutWidget.setGeometry(QRect(24, 30, 291, 80))
        self.gridLayout = QGridLayout(self.gridLayoutWidget)
        self.gridLayout.setObjectName(u"gridLayout")
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.relation = QLabel(self.gridLayoutWidget)
        self.relation.setObjectName(u"relation")

        self.gridLayout.addWidget(self.relation, 0, 1, 1, 1)

        self.Attributes = QLabel(self.gridLayoutWidget)
        self.Attributes.setObjectName(u"Attributes")

        self.gridLayout.addWidget(self.Attributes, 0, 2, 1, 1)

        self.Relation = QLabel(self.gridLayoutWidget)
        self.Relation.setObjectName(u"Relation")

        self.gridLayout.addWidget(self.Relation, 0, 0, 1, 1)

        self.attributes = QLabel(self.gridLayoutWidget)
        self.attributes.setObjectName(u"attributes")

        self.gridLayout.addWidget(self.attributes, 0, 3, 1, 1)

        self.Instances = QLabel(self.gridLayoutWidget)
        self.Instances.setObjectName(u"Instances")

        self.gridLayout.addWidget(self.Instances, 1, 0, 1, 1)

        self.instances = QLabel(self.gridLayoutWidget)
        self.instances.setObjectName(u"instances")

        self.gridLayout.addWidget(self.instances, 1, 1, 1, 1)

        self.Sumofwieghts = QLabel(self.gridLayoutWidget)
        self.Sumofwieghts.setObjectName(u"Sumofwieghts")

        self.gridLayout.addWidget(self.Sumofwieghts, 1, 2, 1, 1)

        self.weights = QLabel(self.gridLayoutWidget)
        self.weights.setObjectName(u"weights")

        self.gridLayout.addWidget(self.weights, 1, 3, 1, 1)

        self.tabWidget.addTab(self.Preprocess, "")
        self.Classify = QWidget()
        self.Classify.setObjectName(u"Classify")
        self.tabWidget.addTab(self.Classify, "")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(MainWindow)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 800, 26))
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName(u"statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        self.openbtn.clicked.connect(MainWindow.openfile)

        self.tabWidget.setCurrentIndex(0)


        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"MainWindow", None))
        self.groupBox.setTitle(QCoreApplication.translate("MainWindow", u"File", None))
        self.openbtn.setText(QCoreApplication.translate("MainWindow", u"Open File", None))
        self.open.setPlaceholderText(QCoreApplication.translate("MainWindow", u"None", None))
        self.groupBox_2.setTitle(QCoreApplication.translate("MainWindow", u"Current relation", None))
        self.relation.setText(QCoreApplication.translate("MainWindow", u"None", None))
        self.Attributes.setText(QCoreApplication.translate("MainWindow", u"Attributes :", None))
        self.Relation.setText(QCoreApplication.translate("MainWindow", u"Relation :", None))
        self.attributes.setText(QCoreApplication.translate("MainWindow", u"None", None))
        self.Instances.setText(QCoreApplication.translate("MainWindow", u"Instances :", None))
        self.instances.setText(QCoreApplication.translate("MainWindow", u"None", None))
        self.Sumofwieghts.setText(QCoreApplication.translate("MainWindow", u"Sum of weights : ", None))
        self.weights.setText(QCoreApplication.translate("MainWindow", u"None", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.Preprocess), QCoreApplication.translate("MainWindow", u"Preprocess", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.Classify), QCoreApplication.translate("MainWindow", u"Classify", None))
    # retranslateUi

