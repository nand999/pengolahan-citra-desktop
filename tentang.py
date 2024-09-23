from PyQt5.QtWidgets import QDialog, QApplication
from PyQt5 import QtCore, QtGui, QtWidgets


class tentangMain(QtWidgets.QDialog): 
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi()

    def setupUi(self):
        self.setObjectName("TentangDialog")
        self.resize(352, 366)
        
        self.listView = QtWidgets.QListView(self)
        self.listView.setGeometry(QtCore.QRect(0, 0, 351, 331))
        self.listView.setObjectName("listView")
        
        self.label = QtWidgets.QLabel(self)
        self.label.setGeometry(QtCore.QRect(40, 40, 261, 31))
        font = QtGui.QFont()
        font.setPointSize(18)
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setAlignment(QtCore.Qt.AlignCenter)  # Menyelaraskan teks ke tengah
        self.label.setObjectName("label")

        
        self.label_2 = QtWidgets.QLabel(self)
        self.label_2.setGeometry(QtCore.QRect(150, 80, 47, 13))
        self.label_2.setObjectName("label_2")
        font = QtGui.QFont()
        font.setPointSize(9)  
        self.label_2.setFont(font)

        self.label_3 = QtWidgets.QLabel(self)
        self.label_3.setGeometry(QtCore.QRect(120, 210, 101, 41))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        
        self.label_4 = QtWidgets.QLabel(self)
        self.label_4.setGeometry(QtCore.QRect(80, 220, 201, 41))
        font = QtGui.QFont()
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.label_4.setFont(font)
        self.label_4.setObjectName("label_4")
        
        self.label_5 = QtWidgets.QLabel(self)
        self.label_5.setGeometry(QtCore.QRect(50, 200, 47, 13))
        self.label_5.setObjectName("label_5")
        
        self.retranslateUi()
        QtCore.QMetaObject.connectSlotsByName(self)

    def retranslateUi(self):
        _translate = QtCore.QCoreApplication.translate
        self.setWindowTitle(_translate("TentangDialog", "Tentang"))
        self.label.setText(_translate("TentangDialog", "PRAKTIKUM PCV"))
        self.label_2.setText(_translate("TentangDialog", "Ver 1.0"))
        self.label_4.setText(_translate("TentangDialog", "KELOMPOK B1"))
        self.label_5.setText(_translate("TentangDialog", "Creator:"))
    

if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    tentang_dialog = tentangMain()
    tentang_dialog.show()
    sys.exit(app.exec_())
