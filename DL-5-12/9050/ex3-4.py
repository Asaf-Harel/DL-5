import sys
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox


class MyMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()  # Calls the constructor of the parent class (QMainWindow) ==> QMainWindow.__init__(self)
        self.setWindowTitle("PyQt5 class")
        self.setGeometry(150, 200, 300, 100)
        self.initUI()

    def initUI(self):
        self.label = QtWidgets.QLabel("Hi there", self)
        self.label.setGeometry(70, 28, 100, 10)

        self.button = QtWidgets.QPushButton("OK", self)
        self.button.setGeometry(10, 20, 50, 30)
        self.button.clicked.connect(self.button_clicked)

    def button_clicked(self):
        self.label.setText("You pressed the button")
        self.label.adjustSize()

        msg = QMessageBox()
        msg.setWindowTitle("My first popup")
        msg.setText("popup message")
        msg.exec_()


app = QApplication(sys.argv)
win = MyMainWindow()
win.show()
sys.exit(app.exec_())
