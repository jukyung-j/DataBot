<<<<<<< HEAD
# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'form.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1011, 784)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.centralwidget)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setObjectName("tabWidget")
        self.Preview = QtWidgets.QWidget()
        self.Preview.setObjectName("Preview")
        self.groupBox_2 = QtWidgets.QGroupBox(self.Preview)
        self.groupBox_2.setGeometry(QtCore.QRect(100, 10, 391, 131))
        self.groupBox_2.setObjectName("groupBox_2")
        self.gridLayoutWidget = QtWidgets.QWidget(self.groupBox_2)
        self.gridLayoutWidget.setGeometry(QtCore.QRect(24, 30, 341, 80))
        self.gridLayoutWidget.setObjectName("gridLayoutWidget")
        self.gridLayout = QtWidgets.QGridLayout(self.gridLayoutWidget)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setObjectName("gridLayout")
        self.relation = QtWidgets.QLabel(self.gridLayoutWidget)
        self.relation.setObjectName("relation")
        self.gridLayout.addWidget(self.relation, 0, 1, 1, 1)
        self.Attributes = QtWidgets.QLabel(self.gridLayoutWidget)
        self.Attributes.setObjectName("Attributes")
        self.gridLayout.addWidget(self.Attributes, 0, 2, 1, 1)
        self.Relation = QtWidgets.QLabel(self.gridLayoutWidget)
        self.Relation.setObjectName("Relation")
        self.gridLayout.addWidget(self.Relation, 0, 0, 1, 1)
        self.attributes = QtWidgets.QLabel(self.gridLayoutWidget)
        self.attributes.setObjectName("attributes")
        self.gridLayout.addWidget(self.attributes, 0, 3, 1, 1)
        self.Instances = QtWidgets.QLabel(self.gridLayoutWidget)
        self.Instances.setObjectName("Instances")
        self.gridLayout.addWidget(self.Instances, 1, 0, 1, 1)
        self.instances = QtWidgets.QLabel(self.gridLayoutWidget)
        self.instances.setObjectName("instances")
        self.gridLayout.addWidget(self.instances, 1, 1, 1, 1)
        self.groupBox_3 = QtWidgets.QGroupBox(self.Preview)
        self.groupBox_3.setGeometry(QtCore.QRect(20, 150, 280, 491))
        self.groupBox_3.setObjectName("groupBox_3")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.groupBox_3)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.list = QtWidgets.QListWidget(self.groupBox_3)
        self.list.setObjectName("list")
        self.horizontalLayout.addWidget(self.list)
        self.groupBox_4 = QtWidgets.QGroupBox(self.Preview)
        self.groupBox_4.setGeometry(QtCore.QRect(510, 10, 451, 391))
        self.groupBox_4.setObjectName("groupBox_4")
        self.value = QtWidgets.QListWidget(self.groupBox_4)
        self.value.setGeometry(QtCore.QRect(20, 50, 161, 321))
        self.value.setObjectName("value")
        self.label = QtWidgets.QLabel(self.groupBox_4)
        self.label.setGeometry(QtCore.QRect(70, 20, 64, 15))
        self.label.setObjectName("label")
        self.des = QtWidgets.QTableWidget(self.groupBox_4)
        self.des.setGeometry(QtCore.QRect(180, 50, 256, 321))
        self.des.setObjectName("des")
        self.des.setColumnCount(0)
        self.des.setRowCount(0)
        self.label_2 = QtWidgets.QLabel(self.groupBox_4)
        self.label_2.setGeometry(QtCore.QRect(260, 20, 64, 15))
        self.label_2.setObjectName("label_2")
        self.openbtn = QtWidgets.QPushButton(self.Preview)
        self.openbtn.setGeometry(QtCore.QRect(10, 50, 75, 41))
        self.openbtn.setObjectName("openbtn")
        self.groupBox_6 = QtWidgets.QGroupBox(self.Preview)
        self.groupBox_6.setGeometry(QtCore.QRect(510, 420, 461, 251))
        self.groupBox_6.setObjectName("groupBox_6")
        self.verticalLayoutWidget = QtWidgets.QWidget(self.groupBox_6)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(20, 29, 421, 201))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.dis_graph = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.dis_graph.setContentsMargins(0, 0, 0, 0)
        self.dis_graph.setObjectName("dis_graph")
        self.tabWidget.addTab(self.Preview, "")
        self.Preprocessing = QtWidgets.QWidget()
        self.Preprocessing.setObjectName("Preprocessing")
        self.groupBox_8 = QtWidgets.QGroupBox(self.Preprocessing)
        self.groupBox_8.setGeometry(QtCore.QRect(90, 480, 751, 201))
        self.groupBox_8.setObjectName("groupBox_8")
        self.verticalLayoutWidget_2 = QtWidgets.QWidget(self.groupBox_8)
        self.verticalLayoutWidget_2.setGeometry(QtCore.QRect(39, 29, 641, 141))
        self.verticalLayoutWidget_2.setObjectName("verticalLayoutWidget_2")
        self.box_graph = QtWidgets.QVBoxLayout(self.verticalLayoutWidget_2)
        self.box_graph.setContentsMargins(0, 0, 0, 0)
        self.box_graph.setObjectName("box_graph")
        self.groupBox_9 = QtWidgets.QGroupBox(self.Preprocessing)
        self.groupBox_9.setGeometry(QtCore.QRect(90, 30, 751, 331))
        self.groupBox_9.setObjectName("groupBox_9")
        self.miss = QtWidgets.QTableWidget(self.groupBox_9)
        self.miss.setGeometry(QtCore.QRect(40, 51, 661, 241))
        self.miss.setObjectName("miss")
        self.miss.setColumnCount(2)
        self.miss.setRowCount(0)
        item = QtWidgets.QTableWidgetItem()
        self.miss.setHorizontalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        self.miss.setHorizontalHeaderItem(1, item)
        self.groupBox_10 = QtWidgets.QGroupBox(self.Preprocessing)
        self.groupBox_10.setGeometry(QtCore.QRect(560, 380, 281, 80))
        self.groupBox_10.setObjectName("groupBox_10")
        self.comboBox_2 = QtWidgets.QComboBox(self.groupBox_10)
        self.comboBox_2.setGeometry(QtCore.QRect(30, 20, 221, 41))
        self.comboBox_2.setObjectName("comboBox_2")
        self.tabWidget.addTab(self.Preprocessing, "")
        self.Classify = QtWidgets.QWidget()
        self.Classify.setObjectName("Classify")
        self.groupBox = QtWidgets.QGroupBox(self.Classify)
        self.groupBox.setGeometry(QtCore.QRect(20, 140, 311, 361))
        self.groupBox.setObjectName("groupBox")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.groupBox)
        self.verticalLayout.setObjectName("verticalLayout")
        self.all_btn = QtWidgets.QPushButton(self.groupBox)
        self.all_btn.setObjectName("all_btn")
        self.verticalLayout.addWidget(self.all_btn)
        self.splitter = QtWidgets.QSplitter(self.groupBox)
        self.splitter.setOrientation(QtCore.Qt.Horizontal)
        self.splitter.setObjectName("splitter")
        self.classi_btn = QtWidgets.QPushButton(self.splitter)
        self.classi_btn.setObjectName("classi_btn")
        self.reg_btn = QtWidgets.QPushButton(self.splitter)
        self.reg_btn.setObjectName("reg_btn")
        self.verticalLayout.addWidget(self.splitter)
        self.knn = QtWidgets.QCheckBox(self.groupBox)
        self.knn.setObjectName("knn")
        self.verticalLayout.addWidget(self.knn)
        self.lr = QtWidgets.QCheckBox(self.groupBox)
        self.lr.setObjectName("lr")
        self.verticalLayout.addWidget(self.lr)
        self.ridge = QtWidgets.QCheckBox(self.groupBox)
        self.ridge.setObjectName("ridge")
        self.verticalLayout.addWidget(self.ridge)
        self.lasso = QtWidgets.QCheckBox(self.groupBox)
        self.lasso.setObjectName("lasso")
        self.verticalLayout.addWidget(self.lasso)
        self.logistic = QtWidgets.QCheckBox(self.groupBox)
        self.logistic.setObjectName("logistic")
        self.verticalLayout.addWidget(self.logistic)
        self.tree = QtWidgets.QCheckBox(self.groupBox)
        self.tree.setObjectName("tree")
        self.verticalLayout.addWidget(self.tree)
        self.apply = QtWidgets.QPushButton(self.groupBox)
        self.apply.setObjectName("apply")
        self.verticalLayout.addWidget(self.apply)
        self.groupBox_5 = QtWidgets.QGroupBox(self.Classify)
        self.groupBox_5.setGeometry(QtCore.QRect(390, 30, 561, 411))
        self.groupBox_5.setObjectName("groupBox_5")
        self.result = QtWidgets.QTableWidget(self.groupBox_5)
        self.result.setGeometry(QtCore.QRect(10, 50, 541, 281))
        self.result.setObjectName("result")
        self.result.setColumnCount(4)
        self.result.setRowCount(0)
        item = QtWidgets.QTableWidgetItem()
        self.result.setHorizontalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        self.result.setHorizontalHeaderItem(1, item)
        item = QtWidgets.QTableWidgetItem()
        self.result.setHorizontalHeaderItem(2, item)
        item = QtWidgets.QTableWidgetItem()
        self.result.setHorizontalHeaderItem(3, item)
        self.groupBox_7 = QtWidgets.QGroupBox(self.Classify)
        self.groupBox_7.setGeometry(QtCore.QRect(10, 30, 321, 80))
        self.groupBox_7.setObjectName("groupBox_7")
        self.comboBox = QtWidgets.QComboBox(self.groupBox_7)
        self.comboBox.setGeometry(QtCore.QRect(20, 40, 271, 22))
        self.comboBox.setObjectName("comboBox")
        self.tabWidget.addTab(self.Classify, "")
        self.horizontalLayout_2.addWidget(self.tabWidget)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1011, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(1)
        self.apply.clicked.connect(MainWindow.apply_algo)
        self.openbtn.clicked.connect(MainWindow.openfile)
        self.all_btn.clicked.connect(MainWindow.algo_all)
        self.classi_btn.clicked.connect(MainWindow.algo_classi)
        self.reg_btn.clicked.connect(MainWindow.algo_reg)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.groupBox_2.setTitle(_translate("MainWindow", "Current relation"))
        self.relation.setText(_translate("MainWindow", "None"))
        self.Attributes.setText(_translate("MainWindow", "Attributes :"))
        self.Relation.setText(_translate("MainWindow", "Relation :"))
        self.attributes.setText(_translate("MainWindow", "None"))
        self.Instances.setText(_translate("MainWindow", "Instances :"))
        self.instances.setText(_translate("MainWindow", "None"))
        self.groupBox_3.setTitle(_translate("MainWindow", "Attributes"))
        self.groupBox_4.setTitle(_translate("MainWindow", "dataset"))
        self.label.setText(_translate("MainWindow", "Value"))
        self.label_2.setText(_translate("MainWindow", "Statics"))
        self.openbtn.setText(_translate("MainWindow", "Open File"))
        self.groupBox_6.setTitle(_translate("MainWindow", "distribution"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.Preview), _translate("MainWindow", "Preview"))
        self.groupBox_8.setTitle(_translate("MainWindow", "Box graph"))
        self.groupBox_9.setTitle(_translate("MainWindow", "Missing value"))
        item = self.miss.horizontalHeaderItem(0)
        item.setText(_translate("MainWindow", "Coulmns"))
        item = self.miss.horizontalHeaderItem(1)
        item.setText(_translate("MainWindow", "Missing"))
        self.groupBox_10.setTitle(_translate("MainWindow", "Method"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.Preprocessing), _translate("MainWindow", "Preprocessing"))
        self.groupBox.setTitle(_translate("MainWindow", "Algorithm"))
        self.all_btn.setText(_translate("MainWindow", "All"))
        self.classi_btn.setText(_translate("MainWindow", "Classification"))
        self.reg_btn.setText(_translate("MainWindow", "Regression"))
        self.knn.setText(_translate("MainWindow", "K-Nearest Neighbor"))
        self.lr.setText(_translate("MainWindow", "Linear Regression"))
        self.ridge.setText(_translate("MainWindow", "Ridge Regression"))
        self.lasso.setText(_translate("MainWindow", "Lasso Regression"))
        self.logistic.setText(_translate("MainWindow", "Logistic Regression"))
        self.tree.setText(_translate("MainWindow", "Decision Tree"))
        self.apply.setText(_translate("MainWindow", "Apply"))
        self.groupBox_5.setTitle(_translate("MainWindow", "Result"))
        item = self.result.horizontalHeaderItem(0)
        item.setText(_translate("MainWindow", "algorithm"))
        item = self.result.horizontalHeaderItem(1)
        item.setText(_translate("MainWindow", "accuracy"))
        item = self.result.horizontalHeaderItem(2)
        item.setText(_translate("MainWindow", "precision"))
        item = self.result.horizontalHeaderItem(3)
        item.setText(_translate("MainWindow", "recall"))
        self.groupBox_7.setTitle(_translate("MainWindow", "Dependent variable"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.Classify), _translate("MainWindow", "Classify"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

=======
# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'form.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1011, 784)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.centralwidget)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setObjectName("tabWidget")
        self.Preview = QtWidgets.QWidget()
        self.Preview.setObjectName("Preview")
        self.groupBox_2 = QtWidgets.QGroupBox(self.Preview)
        self.groupBox_2.setGeometry(QtCore.QRect(100, 10, 391, 131))
        self.groupBox_2.setObjectName("groupBox_2")
        self.gridLayoutWidget = QtWidgets.QWidget(self.groupBox_2)
        self.gridLayoutWidget.setGeometry(QtCore.QRect(24, 30, 341, 80))
        self.gridLayoutWidget.setObjectName("gridLayoutWidget")
        self.gridLayout = QtWidgets.QGridLayout(self.gridLayoutWidget)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setObjectName("gridLayout")
        self.relation = QtWidgets.QLabel(self.gridLayoutWidget)
        self.relation.setObjectName("relation")
        self.gridLayout.addWidget(self.relation, 0, 1, 1, 1)
        self.Attributes = QtWidgets.QLabel(self.gridLayoutWidget)
        self.Attributes.setObjectName("Attributes")
        self.gridLayout.addWidget(self.Attributes, 0, 2, 1, 1)
        self.Relation = QtWidgets.QLabel(self.gridLayoutWidget)
        self.Relation.setObjectName("Relation")
        self.gridLayout.addWidget(self.Relation, 0, 0, 1, 1)
        self.attributes = QtWidgets.QLabel(self.gridLayoutWidget)
        self.attributes.setObjectName("attributes")
        self.gridLayout.addWidget(self.attributes, 0, 3, 1, 1)
        self.Instances = QtWidgets.QLabel(self.gridLayoutWidget)
        self.Instances.setObjectName("Instances")
        self.gridLayout.addWidget(self.Instances, 1, 0, 1, 1)
        self.instances = QtWidgets.QLabel(self.gridLayoutWidget)
        self.instances.setObjectName("instances")
        self.gridLayout.addWidget(self.instances, 1, 1, 1, 1)
        self.groupBox_3 = QtWidgets.QGroupBox(self.Preview)
        self.groupBox_3.setGeometry(QtCore.QRect(20, 150, 280, 491))
        self.groupBox_3.setObjectName("groupBox_3")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.groupBox_3)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.list = QtWidgets.QListWidget(self.groupBox_3)
        self.list.setObjectName("list")
        self.horizontalLayout.addWidget(self.list)
        self.groupBox_4 = QtWidgets.QGroupBox(self.Preview)
        self.groupBox_4.setGeometry(QtCore.QRect(510, 10, 451, 391))
        self.groupBox_4.setObjectName("groupBox_4")
        self.value = QtWidgets.QListWidget(self.groupBox_4)
        self.value.setGeometry(QtCore.QRect(20, 50, 161, 321))
        self.value.setObjectName("value")
        self.label = QtWidgets.QLabel(self.groupBox_4)
        self.label.setGeometry(QtCore.QRect(70, 20, 64, 15))
        self.label.setObjectName("label")
        self.des = QtWidgets.QTableWidget(self.groupBox_4)
        self.des.setGeometry(QtCore.QRect(180, 50, 256, 321))
        self.des.setObjectName("des")
        self.des.setColumnCount(0)
        self.des.setRowCount(0)
        self.label_2 = QtWidgets.QLabel(self.groupBox_4)
        self.label_2.setGeometry(QtCore.QRect(260, 20, 64, 15))
        self.label_2.setObjectName("label_2")
        self.openbtn = QtWidgets.QPushButton(self.Preview)
        self.openbtn.setGeometry(QtCore.QRect(10, 50, 75, 41))
        self.openbtn.setObjectName("openbtn")
        self.groupBox_6 = QtWidgets.QGroupBox(self.Preview)
        self.groupBox_6.setGeometry(QtCore.QRect(510, 420, 461, 251))
        self.groupBox_6.setObjectName("groupBox_6")
        self.verticalLayoutWidget = QtWidgets.QWidget(self.groupBox_6)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(20, 29, 421, 201))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.dis_graph = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.dis_graph.setContentsMargins(0, 0, 0, 0)
        self.dis_graph.setObjectName("dis_graph")
        self.tabWidget.addTab(self.Preview, "")
        self.Preprocessing = QtWidgets.QWidget()
        self.Preprocessing.setObjectName("Preprocessing")
        self.groupBox_8 = QtWidgets.QGroupBox(self.Preprocessing)
        self.groupBox_8.setGeometry(QtCore.QRect(90, 480, 751, 201))
        self.groupBox_8.setObjectName("groupBox_8")
        self.verticalLayoutWidget_2 = QtWidgets.QWidget(self.groupBox_8)
        self.verticalLayoutWidget_2.setGeometry(QtCore.QRect(39, 29, 641, 141))
        self.verticalLayoutWidget_2.setObjectName("verticalLayoutWidget_2")
        self.box_graph = QtWidgets.QVBoxLayout(self.verticalLayoutWidget_2)
        self.box_graph.setContentsMargins(0, 0, 0, 0)
        self.box_graph.setObjectName("box_graph")
        self.groupBox_9 = QtWidgets.QGroupBox(self.Preprocessing)
        self.groupBox_9.setGeometry(QtCore.QRect(90, 30, 751, 331))
        self.groupBox_9.setObjectName("groupBox_9")
        self.miss = QtWidgets.QTableWidget(self.groupBox_9)
        self.miss.setGeometry(QtCore.QRect(40, 51, 661, 241))
        self.miss.setObjectName("miss")
        self.miss.setColumnCount(2)
        self.miss.setRowCount(0)
        item = QtWidgets.QTableWidgetItem()
        self.miss.setHorizontalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        self.miss.setHorizontalHeaderItem(1, item)
        self.groupBox_10 = QtWidgets.QGroupBox(self.Preprocessing)
        self.groupBox_10.setGeometry(QtCore.QRect(560, 380, 281, 80))
        self.groupBox_10.setObjectName("groupBox_10")
        self.comboBox_2 = QtWidgets.QComboBox(self.groupBox_10)
        self.comboBox_2.setGeometry(QtCore.QRect(30, 20, 221, 41))
        self.comboBox_2.setObjectName("comboBox_2")
        self.tabWidget.addTab(self.Preprocessing, "")
        self.Classify = QtWidgets.QWidget()
        self.Classify.setObjectName("Classify")
        self.groupBox = QtWidgets.QGroupBox(self.Classify)
        self.groupBox.setGeometry(QtCore.QRect(20, 140, 311, 361))
        self.groupBox.setObjectName("groupBox")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.groupBox)
        self.verticalLayout.setObjectName("verticalLayout")
        self.all_btn = QtWidgets.QPushButton(self.groupBox)
        self.all_btn.setObjectName("all_btn")
        self.verticalLayout.addWidget(self.all_btn)
        self.splitter = QtWidgets.QSplitter(self.groupBox)
        self.splitter.setOrientation(QtCore.Qt.Horizontal)
        self.splitter.setObjectName("splitter")
        self.classi_btn = QtWidgets.QPushButton(self.splitter)
        self.classi_btn.setObjectName("classi_btn")
        self.reg_btn = QtWidgets.QPushButton(self.splitter)
        self.reg_btn.setObjectName("reg_btn")
        self.verticalLayout.addWidget(self.splitter)
        self.knn = QtWidgets.QCheckBox(self.groupBox)
        self.knn.setObjectName("knn")
        self.verticalLayout.addWidget(self.knn)
        self.lr = QtWidgets.QCheckBox(self.groupBox)
        self.lr.setObjectName("lr")
        self.verticalLayout.addWidget(self.lr)
        self.ridge = QtWidgets.QCheckBox(self.groupBox)
        self.ridge.setObjectName("ridge")
        self.verticalLayout.addWidget(self.ridge)
        self.lasso = QtWidgets.QCheckBox(self.groupBox)
        self.lasso.setObjectName("lasso")
        self.verticalLayout.addWidget(self.lasso)
        self.logistic = QtWidgets.QCheckBox(self.groupBox)
        self.logistic.setObjectName("logistic")
        self.verticalLayout.addWidget(self.logistic)
        self.tree = QtWidgets.QCheckBox(self.groupBox)
        self.tree.setObjectName("tree")
        self.verticalLayout.addWidget(self.tree)
        self.apply = QtWidgets.QPushButton(self.groupBox)
        self.apply.setObjectName("apply")
        self.verticalLayout.addWidget(self.apply)
        self.groupBox_5 = QtWidgets.QGroupBox(self.Classify)
        self.groupBox_5.setGeometry(QtCore.QRect(390, 30, 561, 411))
        self.groupBox_5.setObjectName("groupBox_5")
        self.result = QtWidgets.QTableWidget(self.groupBox_5)
        self.result.setGeometry(QtCore.QRect(10, 50, 541, 281))
        self.result.setObjectName("result")
        self.result.setColumnCount(4)
        self.result.setRowCount(0)
        item = QtWidgets.QTableWidgetItem()
        self.result.setHorizontalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        self.result.setHorizontalHeaderItem(1, item)
        item = QtWidgets.QTableWidgetItem()
        self.result.setHorizontalHeaderItem(2, item)
        item = QtWidgets.QTableWidgetItem()
        self.result.setHorizontalHeaderItem(3, item)
        self.groupBox_7 = QtWidgets.QGroupBox(self.Classify)
        self.groupBox_7.setGeometry(QtCore.QRect(10, 30, 321, 80))
        self.groupBox_7.setObjectName("groupBox_7")
        self.comboBox = QtWidgets.QComboBox(self.groupBox_7)
        self.comboBox.setGeometry(QtCore.QRect(20, 40, 271, 22))
        self.comboBox.setObjectName("comboBox")
        self.tabWidget.addTab(self.Classify, "")
        self.horizontalLayout_2.addWidget(self.tabWidget)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1011, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(1)
        self.apply.clicked.connect(MainWindow.apply_algo)
        self.openbtn.clicked.connect(MainWindow.openfile)
        self.all_btn.clicked.connect(MainWindow.algo_all)
        self.classi_btn.clicked.connect(MainWindow.algo_classi)
        self.reg_btn.clicked.connect(MainWindow.algo_reg)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.groupBox_2.setTitle(_translate("MainWindow", "Current relation"))
        self.relation.setText(_translate("MainWindow", "None"))
        self.Attributes.setText(_translate("MainWindow", "Attributes :"))
        self.Relation.setText(_translate("MainWindow", "Relation :"))
        self.attributes.setText(_translate("MainWindow", "None"))
        self.Instances.setText(_translate("MainWindow", "Instances :"))
        self.instances.setText(_translate("MainWindow", "None"))
        self.groupBox_3.setTitle(_translate("MainWindow", "Attributes"))
        self.groupBox_4.setTitle(_translate("MainWindow", "dataset"))
        self.label.setText(_translate("MainWindow", "Value"))
        self.label_2.setText(_translate("MainWindow", "Statics"))
        self.openbtn.setText(_translate("MainWindow", "Open File"))
        self.groupBox_6.setTitle(_translate("MainWindow", "distribution"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.Preview), _translate("MainWindow", "Preview"))
        self.groupBox_8.setTitle(_translate("MainWindow", "Box graph"))
        self.groupBox_9.setTitle(_translate("MainWindow", "Missing value"))
        item = self.miss.horizontalHeaderItem(0)
        item.setText(_translate("MainWindow", "Coulmns"))
        item = self.miss.horizontalHeaderItem(1)
        item.setText(_translate("MainWindow", "Missing"))
        self.groupBox_10.setTitle(_translate("MainWindow", "Method"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.Preprocessing), _translate("MainWindow", "Preprocessing"))
        self.groupBox.setTitle(_translate("MainWindow", "Algorithm"))
        self.all_btn.setText(_translate("MainWindow", "All"))
        self.classi_btn.setText(_translate("MainWindow", "Classification"))
        self.reg_btn.setText(_translate("MainWindow", "Regression"))
        self.knn.setText(_translate("MainWindow", "K-Nearest Neighbor"))
        self.lr.setText(_translate("MainWindow", "Linear Regression"))
        self.ridge.setText(_translate("MainWindow", "Ridge Regression"))
        self.lasso.setText(_translate("MainWindow", "Lasso Regression"))
        self.logistic.setText(_translate("MainWindow", "Logistic Regression"))
        self.tree.setText(_translate("MainWindow", "Decision Tree"))
        self.apply.setText(_translate("MainWindow", "Apply"))
        self.groupBox_5.setTitle(_translate("MainWindow", "Result"))
        item = self.result.horizontalHeaderItem(0)
        item.setText(_translate("MainWindow", "algorithm"))
        item = self.result.horizontalHeaderItem(1)
        item.setText(_translate("MainWindow", "accuracy"))
        item = self.result.horizontalHeaderItem(2)
        item.setText(_translate("MainWindow", "precision"))
        item = self.result.horizontalHeaderItem(3)
        item.setText(_translate("MainWindow", "recall"))
        self.groupBox_7.setTitle(_translate("MainWindow", "Dependent variable"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.Classify), _translate("MainWindow", "Classify"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

>>>>>>> bae2b6a9b6d11ad486c95d65d14e386e61d25064
