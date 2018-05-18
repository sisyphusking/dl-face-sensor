import sys

from PyQt5 import QtWidgets
from PyQt5 import QtGui


# def show_image(image_path='screen.png'):
#     app = QtWidgets.QApplication(sys.argv)
#     pixmap = QtGui.QPixmap(image_path)
#     screen = QtWidgets.QLabel()
#     screen.setPixmap(pixmap)
#     screen.showFullScreen()
#     sys.exit(app.exec_())


def show_image(image_path='screen.png'):
    app = QtWidgets.QApplication(sys.argv)
    pixmap = QtGui.QPixmap(image_path)
    screen = QtWidgets.QLabel()
    screen.setPixmap(pixmap)
    # screen.showFullScreen()
    screen.showMaximized()
    # screen.showNormal()
    app.exec()
