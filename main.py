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
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

df = pd.DataFrame
X, y, X_train, X_test, y_train, y_test, x_train, x_test =0,0,0,0,0,0,0,0

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
            global X_train, X_test, y_train, y_test  # train 데이터와 test데이터 나누기
            global x_train, x_test

            for attr in df.columns: # 변수들 combobox에 추가
                self.ui.comboBox.addItem(attr)
            result = df.columns[-1]

            # 문자형 데이터 숫자로 바꾸기
            label = LabelEncoder()

            label.fit(df.sex.drop_duplicates())
            df.sex = label.transform(df.sex)
            label.fit(df.smoker.drop_duplicates())
            df.smoker = label.transform(df.smoker)
            label.fit(df.region.drop_duplicates())
            df.region = label.transform(df.region)

            self.ui.comboBox.setCurrentText(result) #default로 마지막 속성 보여주기
            y = df.loc[:, result]
            X = df[df.columns.difference([result])]
            X_train, X_test, y_train, y_test = train_test_split(
                X, y.values.ravel(), random_state=42)

            self.ui.comboBox.currentIndexChanged.connect(self.on_select)    #종속변수가 바뀌면

            file_path = os.path.splitext(fileName)[0]
            file_name = file_path.split('/')[-1]
            self.ui.relation.setText(file_name)
            self.ui.attributes.setText(str(df.shape[1]))
            self.ui.instances.setText(str(len(df)))

            self.ui.list.addItems(df.columns)
            self.ui.value.clear()
            self.ui.list.itemClicked.connect(self.list_click)

            x_train, x_test = X_train, X_test
            # 데이터 표준화(StandardScaler)
            # Standardization 평균 0 / 분산 1
            scaler = StandardScaler()
            scaler.fit(X_train)
            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)

            for i in range(len(df.values[0])):
                if is_string_dtype(df.values[0][i]):
                    print(df.values[0][i])

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
        self.ui.des.setHorizontalHeaderLabels(["describe"])
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
        global X,y,X_train,X_test,y_train,y_test
        y = df.loc[:,result]
        X = df[df.columns.difference([result])]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y.values.ravel(), random_state=42)

    def apply_algo(self):   #알고리즘 적용
        self.ui.result.setRowCount(6)
        self.ui.result.verticalHeader().setVisible(False)
        count = -1
        if self.ui.knn.isChecked() == True:
            count += 1
            self.knn(count)
        if self.ui.lr.isChecked() == True:
            count += 1
            self.linear(count)
        if self.ui.ridge.isChecked() == True:
            count += 1
            self.ridge(count)
        if self.ui.lasso.isChecked() == True:
            count += 1
            self.lasso(count)
        if self.ui.logistic.isChecked() == True:
            count += 1
            self.logistic(count)
        if self.ui.tree.isChecked() == True:
            count += 1
            self.tree(count)

    def knn(self,i):  #knn 알고리즘
        kn = KNeighborsClassifier()
        kn.fit(X_train, y_train)
        y_predict = kn.predict(X_test)
        self.ui.result.setItem(i,0,QTableWidgetItem("Knn"))
        self.ui.result.setItem(i,1,QTableWidgetItem(str(round(accuracy_score(y_test,y_predict),5))))
        self.ui.result.setItem(i,2,QTableWidgetItem(str(round(precision_score(y_test,y_predict,average='weighted'),5))))
        self.ui.result.setItem(i,3,QTableWidgetItem(str(round(recall_score(y_test,y_predict,average='weighted'),5))))

    def linear(self,i):   #linearRegression 알고리즘
        lr = LinearRegression()
        lr.fit(X_train,y_train)
        y_predict = lr.predict(X_test)
        self.ui.result.setItem(i, 0, QTableWidgetItem("LinearRegression"))
        self.ui.result.setItem(i, 1, QTableWidgetItem(str(round(lr.score(X_test,y_test),5))))
        self.ui.result.setItem(i, 2, QTableWidgetItem("RMSE: "+str(round(mean_squared_error(y_test,y_predict)**0.5,5))))
        self.ui.result.setItem(i, 3, QTableWidgetItem("MSE: "+str(round(mean_squared_error(y_test,y_predict),5))))

    def ridge(self, i):     #RassoRegression 알고리즘
        ridge = Ridge(alpha=0.001)
        ridge.fit(X_train, y_train)
        y_predict = ridge.predict(X_test)
        self.ui.result.setItem(i, 0, QTableWidgetItem("RidgeRegression"))
        self.ui.result.setItem(i, 1, QTableWidgetItem(str(round(ridge.score(X_test, y_test), 5))))
        self.ui.result.setItem(i, 2, QTableWidgetItem("RMSE: " + str(round(mean_squared_error(y_test, y_predict)**0.5, 5))))
        self.ui.result.setItem(i, 3, QTableWidgetItem("MSE: " + str(round(mean_squared_error(y_test, y_predict), 5))))

    def lasso(self, i):     #LassoRegression 알고리즘
        lasso = Lasso(alpha=0.001)
        lasso.fit(X_train,y_train)
        y_predict = lasso.predict(X_test)
        self.ui.result.setItem(i, 0, QTableWidgetItem("LassoRegression"))
        self.ui.result.setItem(i, 1, QTableWidgetItem(str(round(lasso.score(X_test, y_test), 5))))
        self.ui.result.setItem(i, 2, QTableWidgetItem("RMSE: " + str(round(mean_squared_error(y_test, y_predict) ** 0.5, 5))))
        self.ui.result.setItem(i, 3, QTableWidgetItem("MSE: " + str(round(mean_squared_error(y_test, y_predict), 5))))

    def logistic(self, i):      #LogisticRegression 알고리즘
        lr = LogisticRegression(C=20, max_iter=1000)
        lr.fit(X_train,y_train)
        y_predict = lr.predict(X_test)
        self.ui.result.setItem(i, 0, QTableWidgetItem("LogisticRegression"))
        self.ui.result.setItem(i, 1, QTableWidgetItem(str(round(accuracy_score(y_test, y_predict), 5))))
        self.ui.result.setItem(i, 2, QTableWidgetItem(str(round(precision_score(y_test,y_predict,average='weighted'), 5))))
        self.ui.result.setItem(i, 3, QTableWidgetItem(str(round(recall_score(y_test,y_predict,average='weighted'), 5))))

    def tree(self, i):      #DecisionTree 알고리즘
        dt = DecisionTreeClassifier(max_depth=5, random_state=42)
        dt.fit(x_train,y_train)  #표준화 하지 않은 데이터
        y_predict = dt.predict(x_test)
        self.ui.result.setItem(i, 0, QTableWidgetItem("DecisionTree"))
        self.ui.result.setItem(i, 1, QTableWidgetItem(str(round(accuracy_score(y_test, y_predict), 5))))
        self.ui.result.setItem(i, 2, QTableWidgetItem(str(round(precision_score(y_test, y_predict, average='weighted'), 5))))
        self.ui.result.setItem(i, 3, QTableWidgetItem(str(round(recall_score(y_test, y_predict, average='weighted'), 5))))

if __name__ == '__main__':
    app = QApplication(sys.argv)

    window = MainWindow()
    window.show()

    sys.exit(app.exec_())