import os
import sys

from PySide6.QtCore import QFile,QIODevice
from PySide6.QtWidgets import *
from form import Ui_MainWindow
from PySide6.QtUiTools import QUiLoader
import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype
from pandas.api.types import is_string_dtype
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
class MainWindow(QMainWindow):

    def __init__(self):
        super(MainWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

    def openfile(self): #파일 열기 버튼 누르면
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(self, "데이터 파일 열기", \
                                                  "", "All Files(*);;CSV data files(*.csv)", options=options)
        if fileName:
            global df
            df = pd.read_csv(fileName)
            global X
            global y
            X = df.iloc[:,:-1]
            y = df.iloc[:,[-1]]
            global X_train, X_test, y_train, y_test  # train 데이터와 test데이터 나누기
            X_train, X_test, y_train, y_test = train_test_split(
                X, y.values.ravel(), random_state=42)

            for attr in df.columns: # 변수들 combobox에 추가
                self.ui.comboBox.addItem(attr)
            result = df.columns[-1]

            self.ui.comboBox.setCurrentText(result) #default로 마지막 속성 보여주기
            self.ui.comboBox.currentIndexChanged.connect(self.on_select)

            file_path = os.path.splitext(fileName)[0]
            file_name = file_path.split('/')[-1]
            self.ui.relation.setText(file_name)
            self.ui.attributes.setText(str(df.shape[1]))
            self.ui.instances.setText(str(len(df)))

            self.ui.list.addItems(df.columns)
            self.ui.value.clear()
            self.ui.list.itemClicked.connect(self.list_click)


    def list_click(self):   # 속성들을 클릭하면
        data = df.loc[:9,[self.ui.list.currentItem().text()]]
        self.ui.value.clear()
        for i in range(10):
            self.ui.value.insertItem(i,QListWidgetItem(str(data.values[i]).strip("[,]")))

        desc =['count','mean','std','min','25%','50%','75%','max']  #데이터가 숫자이면
        desc_str = ['count', 'unique', 'top', 'freq']   #데이터가 문자이면
        self.ui.des.clear()
        self.ui.des.setRowCount(len(desc))
        self.ui.des.setColumnCount(1)
        self.ui.des.setVerticalHeaderLabels(desc)

        if is_numeric_dtype(data.squeeze()) :   #숫자형 데이터면
            for i in range(len(desc)):
                self.ui.des.setItem(i,0,QTableWidgetItem(str(float(data.describe().values[i]))))

        else :  #숫자형 데이터가 아닌경우
            self.ui.des.setRowCount(len(desc_str))
            self.ui.des.setVerticalHeaderLabels(desc_str)
            self.ui.des.setColumnCount(1)
            for i in range(len(desc_str)):
                self.ui.des.setItem(i,0,QTableWidgetItem(str(data.describe().values[i]).strip("[,]")))

    def on_select(self):    #종속변수 정하기
        result = self.ui.comboBox.currentText()
        y = df.loc[:,result]
        X = df[df.columns.difference([result])]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y.values.ravel(), random_state=42)

    def apply_algo(self):   #알고리즘 적용
        self.ui.result.setRowCount(2)
        self.ui.result.verticalHeader().setVisible(False)
        if self.ui.knn.isChecked() == True:
            self.knn()
        if self.ui.lr.isChecked() == True:
            self.linear()

    def knn(self):  #knn 알고리즘
        kn = KNeighborsClassifier()
        kn.fit(X_train, y_train)
        y_predict = kn.predict(X_test)
        self.ui.result.setItem(0,0,QTableWidgetItem("Knn"))
        self.ui.result.setItem(0,1,QTableWidgetItem(str(round(accuracy_score(y_test,y_predict),5))))
        self.ui.result.setItem(0,2,QTableWidgetItem(str(round(precision_score(y_test,y_predict,average='weighted'),5))))
        self.ui.result.setItem(0,3,QTableWidgetItem(str(round(recall_score(y_test,y_predict,average='weighted'),5))))

    def linear(self):   #linearRegression 알고리즘
        lr = LinearRegression()
        lr.fit(X_train,y_train)
        y_predict = lr.predict(X_test)
        self.ui.result.setItem(1, 0, QTableWidgetItem("LinearRegression"))
        self.ui.result.setItem(1, 1, QTableWidgetItem(str(round(lr.score(X_test,y_test),5))))
        self.ui.result.setItem(1, 2, QTableWidgetItem("r2_score: "+str(round(r2_score(y_test,y_predict),5))))
        self.ui.result.setItem(1, 3, QTableWidgetItem("MSE: "+str(round(mean_squared_error(y_test,y_predict),5))))

if __name__ == '__main__':
    app = QApplication(sys.argv)

    window = MainWindow()
    window.show()

    sys.exit(app.exec_())