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
        MainWindow.resize(1011, 784)
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.tabWidget = QTabWidget(self.centralwidget)
        self.tabWidget.setObjectName(u"tabWidget")
        self.tabWidget.setGeometry(QRect(0, 0, 1011, 821))
        self.Preprocess = QWidget()
        self.Preprocess.setObjectName(u"Preprocess")
        self.groupBox = QGroupBox(self.Preprocess)
        self.groupBox.setObjectName(u"groupBox")
        self.groupBox.setGeometry(QRect(40, 30, 861, 71))
        self.openbtn = QPushButton(self.groupBox)
        self.openbtn.setObjectName(u"openbtn")
        self.openbtn.setGeometry(QRect(10, 20, 75, 24))
        self.open = QTextBrowser(self.groupBox)
        self.open.setObjectName(u"open")
        self.open.setGeometry(QRect(100, 20, 631, 31))
        self.open.setCursorWidth(1)
        self.groupBox_2 = QGroupBox(self.Preprocess)
        self.groupBox_2.setObjectName(u"groupBox_2")
        self.groupBox_2.setGeometry(QRect(30, 130, 391, 131))
        self.gridLayoutWidget = QWidget(self.groupBox_2)
        self.gridLayoutWidget.setObjectName(u"gridLayoutWidget")
        self.gridLayoutWidget.setGeometry(QRect(24, 30, 341, 80))
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

        self.groupBox_3 = QGroupBox(self.Preprocess)
        self.groupBox_3.setObjectName(u"groupBox_3")
        self.groupBox_3.setGeometry(QRect(30, 280, 391, 336))
        self.pushButton_3 = QPushButton(self.groupBox_3)
        self.pushButton_3.setObjectName(u"pushButton_3")
        self.pushButton_3.setGeometry(QRect(12, 296, 93, 28))
        self.tableWidget = QTableWidget(self.groupBox_3)
        if (self.tableWidget.columnCount() < 1):
            self.tableWidget.setColumnCount(1)
        __qtablewidgetitem = QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(0, __qtablewidgetitem)
        self.tableWidget.setObjectName(u"tableWidget")
        self.tableWidget.setGeometry(QRect(10, 90, 361, 192))
        self.layoutWidget = QWidget(self.groupBox_3)
        self.layoutWidget.setObjectName(u"layoutWidget")
        self.layoutWidget.setGeometry(QRect(12, 27, 331, 30))
        self.horizontalLayout = QHBoxLayout(self.layoutWidget)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.all = QPushButton(self.layoutWidget)
        self.all.setObjectName(u"all")

        self.horizontalLayout.addWidget(self.all)

        self.none = QPushButton(self.layoutWidget)
        self.none.setObjectName(u"none")

        self.horizontalLayout.addWidget(self.none)

        self.tabWidget.addTab(self.Preprocess, "")
        self.Classify = QWidget()
        self.Classify.setObjectName(u"Classify")
        self.tabWidget.addTab(self.Classify, "")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(MainWindow)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 1011, 26))
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
        self.groupBox_3.setTitle(QCoreApplication.translate("MainWindow", u"Attributes", None))
        self.pushButton_3.setText(QCoreApplication.translate("MainWindow", u"Remove", None))
        ___qtablewidgetitem = self.tableWidget.horizontalHeaderItem(0)
        ___qtablewidgetitem.setText(QCoreApplication.translate("MainWindow", u"Name", None));
        self.all.setText(QCoreApplication.translate("MainWindow", u"All", None))
        self.none.setText(QCoreApplication.translate("MainWindow", u"None", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.Preprocess), QCoreApplication.translate("MainWindow", u"Preprocess", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.Classify), QCoreApplication.translate("MainWindow", u"Classify", None))
    # retranslateUi

