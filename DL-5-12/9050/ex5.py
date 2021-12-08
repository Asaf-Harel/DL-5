import sys
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox, QDialog, QDialogButtonBox


class MyDialog(QDialog):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("My first dialog!")
        self.setGeometry(180, 220, 200, 100)

        self.initUI()

    def initUI(self):
        self.label = QtWidgets.QLabel("Continue?", self)
        self.label.setGeometry(50, 20, 100, 10)

        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)

        self.button_box.setGeometry(100, 70, 100, 30)


app = QApplication(sys.argv)
win = MyDialog()
win.show()
sys.exit(app.exec_())
