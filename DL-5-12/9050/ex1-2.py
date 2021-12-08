import sys
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox

app = QApplication(sys.argv)

win = QMainWindow()
win.setGeometry(150, 200, 300, 40)
win.setWindowTitle("my first PyQt5 window")


def button_clicked():
    msg = QMessageBox()
    msg.setWindowTitle("My first popup")
    msg.setText("Hi There")
    msg.exec_()


label = QtWidgets.QLabel("Hi there", win)

button = QtWidgets.QPushButton("OK", win)
button.clicked.connect(button_clicked)

win.show()
sys.exit(app.exec_())
