import os   # 운영체제(OS : Operating System)를 제어
import sys  # 파이썬 인터프리터를 제어할 수 있는 방법

# PyQt5관련 헤더
from PyQt5.QtCore import QFile,QIODevice
from PyQt5.QtWidgets import *
from PyQt5 import uic

import pandas as pd # 데이터 프레임과 시리즈를 사용하기 쉽게
import numpy as np  # 수학적 연산을 쉽게

import matplotlib.pyplot as plt  # 차트그리기

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas    # 그래프그리기

from pandas.api.types import is_numeric_dtype, is_string_dtype   # 선택된 속성이 숫자형 문자형인지 알아내기 위해
from sklearn.model_selection import train_test_split, GridSearchCV,cross_validate    # train데이터와 test데이터 분리, 하이퍼파라미터 튜닝
from sklearn.preprocessing import LabelEncoder, StandardScaler  # 문자형데이터를 숫자로 바꾸기 위해, 데이터 표준화
from sklearn.impute import SimpleImputer    # 결측치 처리방법

from sklearn.metrics import accuracy_score, recall_score, precision_score, mean_squared_error   # 분류모델에서 성능평가 지표인 정확도, 재현율, 정밀도 계산, 회귀모델에서 성능평가 지표인 MSE 계산

from sklearn.neighbors import KNeighborsClassifier  # Knn 알고리즘 
from sklearn.linear_model import LinearRegression, Lasso, Ridge, LogisticRegression   # Linear Regression, Lasso Regression, Ridge Regression, Logistic Regression 알고리즘
from sklearn.tree import DecisionTreeClassifier     # Decision Tree 알고리즘

df = pd.DataFrame
X, y, X_train, X_test, y_train, y_test, x_train, x_test =0,0,0,0,0,0,0,0
form_class = uic.loadUiType("form.ui")[0]
dt_class = uic.loadUiType("dt.ui")[0]

class MainWindow(QMainWindow, form_class):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setupUi(self)
        self.fig = plt.Figure()
        self.canvas = FigureCanvas(self.fig)
        self.setWindowTitle("DataBot")

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

            # 초기화
            self.list.clear()
            self.comboBox.clear()
            self.method.clear()
            self.result.clear()
            self.comboBox.blockSignals(True) # 파일 다시 열었을 때 충돌 해결
            self.method.blockSignals(True)
            self.k_fold.clear()

            for attr in df.columns: # 변수들 combobox에 추가
                self.comboBox.addItem(attr)
            result = df.columns[-1]

            self.comboBox.setCurrentText(result) #default로 마지막 속성 보여주기
            y = df.iloc[:, -1]
            X = df.iloc[:,:-1]
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, random_state=42)

            self.comboBox.currentIndexChanged.connect(self.on_select)    #종속변수가 바뀌면

            file_path = os.path.splitext(fileName)[0]
            file_name = file_path.split('/')[-1]
            self.relation.setText(file_name)
            self.attributes.setText(str(df.shape[1]))
            self.instances.setText(str(len(df)))

            self.list.addItems(df.columns)
            self.value.clear()
            self.list.itemClicked.connect(self.list_click)

            self.k_fold.addItems(["","5","10"])

            x_train, x_test = X_train, X_test

            # 데이터 전처리
            column_length = df.shape[1]
            self.miss.setRowCount(column_length)
            for i in range(column_length):
                self.miss.setItem(i, 0, QTableWidgetItem(str(df.columns[i])))
                self.miss.setItem(i, 1, QTableWidgetItem(str(df.isnull().sum()[i])))
            self.method.addItem("delete")
            self.method.addItem("mean")
            self.method.addItem("median")
            self.method.addItem("most_frequent")

            # Box graph
            self.box_graph.addWidget(self.canvas)
            self.fig.clear()

            ax = self.fig.add_subplot(111)
            ax.boxplot(X)
            ax.set_xticklabels(X)

            self.canvas.draw()

            # 데이터 표준화(StandardScaler)
            # Standardization 평균 0 / 분산 1
            scaler = StandardScaler()
            scaler.fit(X_train)
            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)


    def list_click(self):   # 속성들을 클릭하면
        curr_data = self.list.currentItem().text()
        data = df.loc[:10,[curr_data]]
        self.value.clear()
        for i in range(10):
            self.value.insertItem(i,QListWidgetItem(str(data.values[i]).strip("[,]")))
        desc =['count','mean','std','min','25%','50%','75%','max']  #데이터가 숫자이면
        desc_str = ['count', 'unique', 'top', 'freq']   #데이터가 문자이면
        self.des.clear()
        self.des.setRowCount(len(desc))
        self.des.setColumnCount(1)
        self.des.setHorizontalHeaderLabels(["describe"])
        self.des.setVerticalHeaderLabels(desc)

        if is_numeric_dtype(data.squeeze()) :   #숫자형 데이터면
            for i in range(len(desc)):
                self.des.setItem(i,0,QTableWidgetItem(str(float(data.describe().values[i]))))

        else :  #숫자형 데이터가 아닌경우
            self.des.setRowCount(len(desc_str))
            self.des.setVerticalHeaderLabels(desc_str)
            self.des.setColumnCount(1)
            for i in range(len(desc_str)):
                self.des.setItem(i,0,QTableWidgetItem(str(data.describe().values[i]).strip("[,]")))

        self.des.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)

        # 그래프그리기

        self.dis_graph.addWidget(self.canvas)
        self.fig.clear()

        x = df.loc[:, [curr_data]]
        ax = self.fig.add_subplot(111)
        ax.hist(x,histtype='step')

        self.canvas.draw()

    # 결측치 처리 방법
    def method_apply(self):
        global df,X,y,X_train,X_test,y_train,y_test,x_train, x_test
        if self.method.currentText() == 'delete':   # 결측치 제거
            df = df.dropna()

        elif self.method.currentText() == 'mean':   # 평균값
            imputer = SimpleImputer(strategy="mean")
            df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

        elif self.method.currentText() == 'median':  # 중간값
            imputer = SimpleImputer(strategy="median")
            df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

        elif self.method.currentText() == 'most_frequent':   # 최빈값
            imputer = SimpleImputer(strategy="most_frequent")
            df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
        y = df.iloc[:, -1]
        X = df.iloc[:, :-1]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, random_state=42)

        x_train, x_test = X_train, X_test   # Decision Tree

        # Missing Value 다시보여주기
        column_length = df.shape[1]
        for i in range(column_length):
            self.miss.setItem(i, 1, QTableWidgetItem(str(df.isnull().sum()[i])))

        # 박스플롯 다시그리기
        self.fig.clear()

        ax = self.fig.add_subplot(111)
        ax.boxplot(X)
        ax.set_xticklabels(X)

        self.canvas.draw()

    def on_select(self):    #종속변수 정하기
        result = self.comboBox.currentText()
        global X,y,X_train,X_test,y_train,y_test
        y = df.loc[:,result]
        X = df[df.columns.difference([result])]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y.values.ravel(), random_state=42)

    def algo_all(self): # 알고리즘 All 버튼 클릭
        self.knn.toggle()
        self.lr.toggle()
        self.ridge.toggle()
        self.lasso.toggle()
        self.logistic.toggle()
        self.tree.toggle()

    def algo_classi(self):  # 알고리즘 classification 버튼 클릭
        self.knn.toggle()
        self.logistic.toggle()
        self.tree.toggle()

    def algo_reg(self): # 알고리즘 regression 버튼 클릭
        self.lr.toggle()
        self.ridge.toggle()
        self.lasso.toggle()
        self.tree.toggle()


    def apply_algo(self):   #알고리즘 적용
        global X_train,X_test, x_train, x_test
        self.result.clear()
        self.result.setHorizontalHeaderLabels(["algorithm",'params','accuracy','precision','recall'])

        count = -1

        k = int(self.k_fold.currentText()) if self.k_fold.currentText() is not "" else 0   # kfold

        if k is not 0:  # 교차검증할 k가 0이 아닐때
            if self.knn.isChecked() == True: # Knn 알고리즘
                count += 1

                kn = KNeighborsClassifier()

                params = {'n_neighbors' : [1, 3, 5, 7, 9, 11]}  # 하이퍼파라미터
                kn_models = GridSearchCV(kn, params, cv=k, n_jobs=-1)
                kn_models.fit(X_train,y_train)
                knn = kn_models.best_estimator_
                y_pred = knn.predict(X_test)

                self.result.setItem(count, 0, QTableWidgetItem("Knn"))
                self.result.setItem(count, 1, QTableWidgetItem(str(kn_models.best_params_)))
                self.result.setItem(count, 2, QTableWidgetItem(str(round(accuracy_score(y_test, y_pred), 5))))
                self.result.setItem(count, 3, QTableWidgetItem(str(round(precision_score(y_test, y_pred, average='weighted'), 5))))
                self.result.setItem(count, 4, QTableWidgetItem(str(round(recall_score(y_test, y_pred, average='weighted'), 5))))

            if self.lr.isChecked() == True: # Linear Regression 알고리즘
                count += 1

                lr = LinearRegression()
                lr.fit(X_train, y_train)
                y_predict = lr.predict(X_test)
                self.result.setItem(count, 0, QTableWidgetItem("LinearRegression"))
                self.result.setItem(count, 2, QTableWidgetItem(str(round(lr.score(X_test, y_test), 5))))
                self.result.setItem(count, 3, QTableWidgetItem("RMSE: " + str(round(mean_squared_error(y_test, y_predict) ** 0.5, 5))))
                self.result.setItem(count, 4, QTableWidgetItem("MSE: " + str(round(mean_squared_error(y_test, y_predict), 5))))

            if self.ridge.isChecked() == True:  # Ridge Regression 알고리즘
                count += 1

                ridge = Ridge()
                alphas = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
                params = {'alpha': alphas}  # 하이퍼파라미터
                ri_models = GridSearchCV(ridge, params, cv=k, n_jobs=-1)
                ri_models.fit(X_train, y_train)
                ri = ri_models.best_estimator_
                y_pred = ri.predict(X_test)

                self.result.setItem(count, 0, QTableWidgetItem("RidgeRegression"))
                self.result.setItem(count, 1, QTableWidgetItem(str(ri_models.best_params_)))
                self.result.setItem(count, 2, QTableWidgetItem(str(round(ri.score(X_test, y_test), 5))))
                self.result.setItem(count, 3, QTableWidgetItem("RMSE: " + str(round(mean_squared_error(y_test, y_pred) ** 0.5, 5))))
                self.result.setItem(count, 4, QTableWidgetItem("MSE: " + str(round(mean_squared_error(y_test, y_pred), 5))))

            if self.lasso.isChecked() == True:  # Lasso Regression 알고리즘
                count += 1

                lasso = Lasso()
                alphas = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
                params = {'alpha': alphas}  # 하이퍼파라미터
                la_models = GridSearchCV(lasso, params, cv=k, n_jobs=-1)
                la_models.fit(X_train, y_train)
                la = la_models.best_estimator_
                y_pred = la.predict(X_test)

                # print(round(la_models.best_params_,5))
                self.result.setItem(count, 0, QTableWidgetItem("LassoRegression"))
                self.result.setItem(count, 1, QTableWidgetItem(str(la_models.best_params_)))
                self.result.setItem(count, 2, QTableWidgetItem(str(round(la.score(X_test, y_test), 5))))
                self.result.setItem(count, 3, QTableWidgetItem("RMSE: " + str(round(mean_squared_error(y_test, y_pred) ** 0.5, 5))))
                self.result.setItem(count, 4, QTableWidgetItem("MSE: " + str(round(mean_squared_error(y_test, y_pred), 5))))

            if self.logistic.isChecked() == True:   # Logistic Regression 앍고리즘
                count += 1

                params = {'C':[0.001, 0.01, 0.1, 1, 10, 100],
                          'max_iter':[100,1000]}
                lgr = LogisticRegression()
                lgr_models = GridSearchCV(lgr, params, cv=k, n_jobs=-1)
                lgr_models.fit(X_train, y_train)
                lr = lgr_models.best_estimator_
                y_pred = lr.predict(X_test)

                self.result.setItem(count, 0, QTableWidgetItem("LogisticRegression"))
                self.result.setItem(count, 1, QTableWidgetItem(str(lgr_models.best_params_)))
                self.result.setItem(count, 2, QTableWidgetItem(str(round(accuracy_score(y_test, y_pred), 5))))
                self.result.setItem(count, 3, QTableWidgetItem(str(round(precision_score(y_test, y_pred, average='weighted'), 5))))
                self.result.setItem(count, 4, QTableWidgetItem(str(round(recall_score(y_test, y_pred, average='weighted'), 5))))

            if self.tree.isChecked() == True:   # Decision Tree 알고리즘
                count += 1

                self.second = SubWindow()
                self.second.showModal()
                params = self.second.params

                dt = DecisionTreeClassifier()
                dt_models = GridSearchCV(dt, params, cv=k, n_jobs=-1)
                dt_models.fit(x_train, y_train)
                dte = dt_models.best_estimator_
                y_pred = dte.predict(x_test)

                self.result.setItem(count, 0, QTableWidgetItem("DecisionTree"))
                self.result.setItem(count, 1, QTableWidgetItem(str(params)))
                self.result.setItem(count, 2, QTableWidgetItem(str(round(accuracy_score(y_test, y_pred), 5))))
                self.result.setItem(count, 3, QTableWidgetItem(str(round(precision_score(y_test, y_pred, average='weighted'), 5))))
                self.result.setItem(count, 4, QTableWidgetItem(str(round(recall_score(y_test, y_pred, average='weighted'), 5))))
                self.result.setItem(count+1, 3, QTableWidgetItem("RMSE: " + str(round(mean_squared_error(y_test, y_pred) ** 0.5, 5))))
                self.result.setItem(count+1, 4, QTableWidgetItem("MSE: " + str(round(mean_squared_error(y_test, y_pred), 5))))

        else:   # 교차검증을 안할 때
            if self.knn.isChecked() == True:  # Knn 알고리즘
                count += 1

                kn = KNeighborsClassifier()
                kn.fit(X_train, y_train)
                y_predict = kn.predict(X_test)
                self.result.setItem(count, 0, QTableWidgetItem("Knn"))
                self.result.setItem(count, 2, QTableWidgetItem(str(round(accuracy_score(y_test, y_predict), 5))))
                self.result.setItem(count, 3, QTableWidgetItem(
                    str(round(precision_score(y_test, y_predict, average='weighted'), 5))))
                self.result.setItem(count, 4, QTableWidgetItem(
                    str(round(recall_score(y_test, y_predict, average='weighted'), 5))))

            if self.lr.isChecked() == True:  # Linear Regression 알고리즘
                count += 1

                lr = LinearRegression()
                lr.fit(X_train, y_train)
                y_predict = lr.predict(X_test)
                self.result.setItem(count, 0, QTableWidgetItem("LinearRegression"))
                self.result.setItem(count, 2, QTableWidgetItem(str(round(lr.score(X_test, y_test), 5))))
                self.result.setItem(count, 3, QTableWidgetItem(
                    "RMSE: " + str(round(mean_squared_error(y_test, y_predict) ** 0.5, 5))))
                self.result.setItem(count, 4,
                                    QTableWidgetItem("MSE: " + str(round(mean_squared_error(y_test, y_predict), 5))))

            if self.ridge.isChecked() == True:  # Ridge Regression 알고리즘
                count += 1

                ridge = Ridge(alpha=0.001)
                ridge.fit(X_train, y_train)
                y_predict = ridge.predict(X_test)
                self.result.setItem(count, 0, QTableWidgetItem("RidgeRegression"))
                self.result.setItem(count, 2, QTableWidgetItem(str(round(ridge.score(X_test, y_test), 5))))
                self.result.setItem(count, 3, QTableWidgetItem(
                    "RMSE: " + str(round(mean_squared_error(y_test, y_predict) ** 0.5, 5))))
                self.result.setItem(count, 4,
                                    QTableWidgetItem("MSE: " + str(round(mean_squared_error(y_test, y_predict), 5))))

            if self.lasso.isChecked() == True:  # Lasso Regression 알고리즘
                count += 1

                lasso = Lasso(alpha=0.001)
                lasso.fit(X_train, y_train)
                y_predict = lasso.predict(X_test)
                self.result.setItem(count, 0, QTableWidgetItem("LassoRegression"))
                self.result.setItem(count, 2, QTableWidgetItem(str(round(lasso.score(X_test, y_test), 5))))
                self.result.setItem(count, 3, QTableWidgetItem(
                    "RMSE: " + str(round(mean_squared_error(y_test, y_predict) ** 0.5, 5))))
                self.result.setItem(count, 4,
                                    QTableWidgetItem("MSE: " + str(round(mean_squared_error(y_test, y_predict), 5))))

            if self.logistic.isChecked() == True:  # Logistic Regression 앍고리즘
                count += 1

                lr = LogisticRegression(C=20, max_iter=1000)
                lr.fit(X_train, y_train)
                y_predict = lr.predict(X_test)
                self.result.setItem(count, 0, QTableWidgetItem("LogisticRegression"))
                self.result.setItem(count, 2, QTableWidgetItem(str(round(accuracy_score(y_test, y_predict), 5))))
                self.result.setItem(count, 3, QTableWidgetItem(
                    str(round(precision_score(y_test, y_predict, average='weighted'), 5))))
                self.result.setItem(count, 4, QTableWidgetItem(
                    str(round(recall_score(y_test, y_predict, average='weighted'), 5))))

            if self.tree.isChecked() == True:  # Decision Tree 알고리즘
                count += 1
                self.second = SubWindow()
                self.second.showModal()
                params = self.second.p

                dt = DecisionTreeClassifier(splitter=params['splitter'], max_depth=params['max_depth'], min_samples_split=params['min_samples_splits'],
                                    min_samples_leaf=params['min_samples_leaf'], min_weight_fraction_leaf=params['min_weight'],
                                    max_features=params['max_features'], max_leaf_nodes=params['max_leaf'], min_impurity_decrease=params['min_impurity'],
                                    random_state=params['seed'])
                dt.fit(x_train, y_train)  # 표준화 하지 않은 데이터
                y_pred = dt.predict(x_test)

                self.result.setItem(count, 0, QTableWidgetItem("DecisionTree"))
                self.result.setItem(count, 1, QTableWidgetItem(str(params)))
                self.result.setItem(count, 2, QTableWidgetItem(str(round(accuracy_score(y_test, y_pred), 5))))
                self.result.setItem(count, 3, QTableWidgetItem(
                    str(round(precision_score(y_test, y_pred, average='weighted'), 5))))
                self.result.setItem(count, 4,
                                    QTableWidgetItem(str(round(recall_score(y_test, y_pred, average='weighted'), 5))))
                self.result.setItem(count + 1, 3, QTableWidgetItem(
                    "RMSE: " + str(round(mean_squared_error(y_test, y_pred) ** 0.5, 5))))
                self.result.setItem(count + 1, 4,
                                    QTableWidgetItem("MSE: " + str(round(mean_squared_error(y_test, y_pred), 5))))

        self.result.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)

class SubWindow(QDialog, dt_class):     # Decision Tree Setting
    def __init__(self):
        super(QDialog, self).__init__()
        self.setupUi(self)
        self.setWindowTitle("Decision Tree Setting")
        self.splitter.setChecked(True)  # splitter 기본으로 best

        self.params = None
        self.p =None

    def dt_apply(self):
        splitter = "best" if self.splitter.isChecked else "random"

        # spinbox 최솟값 설정
        self.max_depth.setMinimum(1)
        self.min_samples_splits.setMinimum(2)
        self.min_samples_leaf.setMinimum(1)
        self.min_weight.setRange(0.0,0.5)
        self.max_features.setMinimum(1)
        self.max_leaf.setMinimum(2)
        self.min_impurity.setMinimum(0.0)

        p = dict()
        # 값 가져오기
        p['splitter'] = splitter
        p['max_depth'] = self.max_depth.value()
        p['min_samples_splits'] = self.min_samples_splits.value()
        p['min_samples_leaf'] = self.min_samples_leaf.value()
        p['min_weight'] = self.min_weight.value()
        p['max_features'] = self.max_features.value()
        p['seed'] = self.seed.value()
        p['max_leaf'] = self.max_leaf.value()
        p['min_impurity'] = self.min_impurity.value()

        params={'splitter':[p['splitter']], 'max_depth':[p['max_depth']], 'min_samples_split':[p['min_samples_splits']],
                                    'min_samples_leaf':[p['min_samples_leaf']], 'min_weight_fraction_leaf':[p['min_weight']],
                                    'max_features':[p['max_features']], 'max_leaf_nodes':[p['max_leaf']], 'min_impurity_decrease':[p['min_impurity']],
                                    'random_state':[p['seed']]}

        self.params = params
        self.p = p
        self.close()

    def showModal(self):
        return super().exec_()

if __name__ == '__main__':
    app = QApplication(sys.argv)

    window = MainWindow()
    window.show()

    sys.exit(app.exec_())
