import os
import sys

from PySide6.QtCore import QFile,QIODevice
from PySide6.QtWidgets import *
from form import Ui_MainWindow
from PySide6.QtUiTools import QUiLoader
import pandas as pd

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
            df = pd.read_csv(fileName)
            file_path = os.path.splitext(fileName)[0]
            file_name = file_path.split('/')[-1]
            self.ui.open.setText(file_name)
            self.ui.relation.setText(file_name)
            self.ui.attributes.setText(str(df.shape[1]))
            self.ui.instances.setText(str(len(df)))

if __name__ == '__main__':
    app = QApplication(sys.argv)

    window = MainWindow()
    window.show()

    sys.exit(app.exec_())