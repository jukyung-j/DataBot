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
        self.horizontalLayout_2 = QHBoxLayout(self.centralwidget)
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.tabWidget = QTabWidget(self.centralwidget)
        self.tabWidget.setObjectName(u"tabWidget")
        self.Preprocess = QWidget()
        self.Preprocess.setObjectName(u"Preprocess")
        self.groupBox_2 = QGroupBox(self.Preprocess)
        self.groupBox_2.setObjectName(u"groupBox_2")
        self.groupBox_2.setGeometry(QRect(100, 10, 391, 131))
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
        self.groupBox_3.setGeometry(QRect(20, 150, 280, 491))
        self.horizontalLayout = QHBoxLayout(self.groupBox_3)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.list = QListWidget(self.groupBox_3)
        self.list.setObjectName(u"list")

        self.horizontalLayout.addWidget(self.list)

        self.groupBox_4 = QGroupBox(self.Preprocess)
        self.groupBox_4.setObjectName(u"groupBox_4")
        self.groupBox_4.setGeometry(QRect(510, 10, 451, 411))
        self.value = QListWidget(self.groupBox_4)
        self.value.setObjectName(u"value")
        self.value.setGeometry(QRect(20, 50, 161, 321))
        self.label = QLabel(self.groupBox_4)
        self.label.setObjectName(u"label")
        self.label.setGeometry(QRect(70, 20, 64, 15))
        self.des = QTableWidget(self.groupBox_4)
        self.des.setObjectName(u"des")
        self.des.setGeometry(QRect(180, 50, 256, 321))
        self.label_2 = QLabel(self.groupBox_4)
        self.label_2.setObjectName(u"label_2")
        self.label_2.setGeometry(QRect(260, 20, 64, 15))
        self.openbtn = QPushButton(self.Preprocess)
        self.openbtn.setObjectName(u"openbtn")
        self.openbtn.setGeometry(QRect(10, 20, 75, 41))
        self.tabWidget.addTab(self.Preprocess, "")
        self.Classify = QWidget()
        self.Classify.setObjectName(u"Classify")
        self.groupBox = QGroupBox(self.Classify)
        self.groupBox.setObjectName(u"groupBox")
        self.groupBox.setGeometry(QRect(20, 40, 280, 401))
        self.verticalLayout = QVBoxLayout(self.groupBox)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.knn = QCheckBox(self.groupBox)
        self.knn.setObjectName(u"knn")

        self.verticalLayout.addWidget(self.knn)

        self.lr = QCheckBox(self.groupBox)
        self.lr.setObjectName(u"lr")

        self.verticalLayout.addWidget(self.lr)

        self.apply = QPushButton(self.groupBox)
        self.apply.setObjectName(u"apply")

        self.verticalLayout.addWidget(self.apply)

        self.groupBox_5 = QGroupBox(self.Classify)
        self.groupBox_5.setObjectName(u"groupBox_5")
        self.groupBox_5.setGeometry(QRect(390, 30, 561, 411))
        self.tableWidget = QTableWidget(self.groupBox_5)
        if (self.tableWidget.columnCount() < 4):
            self.tableWidget.setColumnCount(4)
        __qtablewidgetitem = QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(0, __qtablewidgetitem)
        __qtablewidgetitem1 = QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(1, __qtablewidgetitem1)
        __qtablewidgetitem2 = QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(2, __qtablewidgetitem2)
        __qtablewidgetitem3 = QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(3, __qtablewidgetitem3)
        self.tableWidget.setObjectName(u"tableWidget")
        self.tableWidget.setGeometry(QRect(20, 50, 521, 281))
        self.tabWidget.addTab(self.Classify, "")

        self.horizontalLayout_2.addWidget(self.tabWidget)

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
        self.apply.clicked.connect(MainWindow.apply_algo)

        self.tabWidget.setCurrentIndex(1)


        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"MainWindow", None))
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
        self.groupBox_4.setTitle(QCoreApplication.translate("MainWindow", u"dataset", None))
        self.label.setText(QCoreApplication.translate("MainWindow", u"Value", None))
        self.label_2.setText(QCoreApplication.translate("MainWindow", u"Statics", None))
        self.openbtn.setText(QCoreApplication.translate("MainWindow", u"Open File", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.Preprocess), QCoreApplication.translate("MainWindow", u"Preprocess", None))
        self.groupBox.setTitle(QCoreApplication.translate("MainWindow", u"Algorithm", None))
        self.knn.setText(QCoreApplication.translate("MainWindow", u"K-Nearest Neighbor", None))
        self.lr.setText(QCoreApplication.translate("MainWindow", u"Logistic Regression", None))
        self.apply.setText(QCoreApplication.translate("MainWindow", u"Apply", None))
        self.groupBox_5.setTitle(QCoreApplication.translate("MainWindow", u"Result", None))
        ___qtablewidgetitem = self.tableWidget.horizontalHeaderItem(0)
        ___qtablewidgetitem.setText(QCoreApplication.translate("MainWindow", u"algorithm", None));
        ___qtablewidgetitem1 = self.tableWidget.horizontalHeaderItem(1)
        ___qtablewidgetitem1.setText(QCoreApplication.translate("MainWindow", u"accuracy", None));
        ___qtablewidgetitem2 = self.tableWidget.horizontalHeaderItem(2)
        ___qtablewidgetitem2.setText(QCoreApplication.translate("MainWindow", u"precision", None));
        ___qtablewidgetitem3 = self.tableWidget.horizontalHeaderItem(3)
        ___qtablewidgetitem3.setText(QCoreApplication.translate("MainWindow", u"recall", None));
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.Classify), QCoreApplication.translate("MainWindow", u"Classify", None))
    # retranslateUi

