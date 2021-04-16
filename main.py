import os
import sys

from PySide6.QtCore import QFile,QIODevice
from PySide6.QtWidgets import *
from form import Ui_MainWindow
from PySide6.QtUiTools import QUiLoader
import pandas as pd
from pandas.api.types import is_numeric_dtype
from pandas.api.types import is_string_dtype
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score

class MainWindow(QMainWindow):

    def __init__(self):
        super(MainWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

    def openfile(self):
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
            file_path = os.path.splitext(fileName)[0]
            file_name = file_path.split('/')[-1]
            self.ui.relation.setText(file_name)
            self.ui.attributes.setText(str(df.shape[1]))
            self.ui.instances.setText(str(len(df)))

            self.ui.list.addItems(df.columns)
            self.ui.value.clear()
            self.ui.list.itemClicked.connect(self.list_click)


    def list_click(self):
        data = df.loc[:9,[self.ui.list.currentItem().text()]]
        self.ui.value.clear()
        for i in range(10):
            self.ui.value.insertItem(i,QListWidgetItem(str(data.values[i]).strip("[,]")))

        desc =['count','mean','std','min','25%','50%','75%','max']
        desc_str = ['count', 'unique', 'top', 'freq']
        self.ui.des.clear()
        self.ui.des.setRowCount(len(desc))
        self.ui.des.setColumnCount(1)
        self.ui.des.setVerticalHeaderLabels(desc)

        if is_numeric_dtype(data.squeeze()) :
            for i in range(len(desc)):
                self.ui.des.setItem(i,0,QTableWidgetItem(str(float(data.describe().values[i]))))

        else :
            self.ui.des.setRowCount(len(desc_str))
            self.ui.des.setVerticalHeaderLabels(desc_str)
            self.ui.des.setColumnCount(1)
            for i in range(len(desc_str)):
                self.ui.des.setItem(i,0,QTableWidgetItem(str(data.describe().values[i]).strip("[,]")))

    def apply_algo(self):
        self.ui.result.setRowCount(2)
        self.ui.result.verticalHeader().setVisible(False)
        if self.ui.knn.isChecked() == True:
            self.knn()
        if self.ui.lr.isChecked() == True:
            self.linear()

    def knn(self):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y.values.ravel(), random_state=42)
        kn = KNeighborsClassifier()
        kn.fit(X_train, y_train)
        y_predict = kn.predict(X_test)
        self.ui.result.setItem(0,0,QTableWidgetItem("Knn"))
        self.ui.result.setItem(0,1,QTableWidgetItem(str(accuracy_score(y_test,y_predict))))
        self.ui.result.setItem(0,2,QTableWidgetItem(str(precision_score(y_test,y_predict,average='weighted'))))
        self.ui.result.setItem(0,3,QTableWidgetItem(str(recall_score(y_test,y_predict,average='weighted'))))

    def linear(self):
        print("lr")

if __name__ == '__main__':
    app = QApplication(sys.argv)

    window = MainWindow()
    window.show()

    sys.exit(app.exec_())