# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ui/connection_type.ui'
#
# Created by: PyQt4 UI code generator 4.11.4
#
# WARNING! All changes made in this file will be lost!

from PyQt4 import QtCore, QtGui

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s

try:
    _encoding = QtGui.QApplication.UnicodeUTF8
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig, _encoding)
except AttributeError:
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig)

class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName(_fromUtf8("Dialog"))
        Dialog.resize(544, 474)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(_fromUtf8(":/icons/images/editresize.png")), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        Dialog.setWindowIcon(icon)
        self.verticalLayout = QtGui.QVBoxLayout(Dialog)
        self.verticalLayout.setObjectName(_fromUtf8("verticalLayout"))
        self.formLayout_3 = QtGui.QFormLayout()
        self.formLayout_3.setLabelAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.formLayout_3.setObjectName(_fromUtf8("formLayout_3"))
        self.label = QtGui.QLabel(Dialog)
        self.label.setObjectName(_fromUtf8("label"))
        self.formLayout_3.setWidget(0, QtGui.QFormLayout.LabelRole, self.label)
        self.label_2 = QtGui.QLabel(Dialog)
        self.label_2.setObjectName(_fromUtf8("label_2"))
        self.formLayout_3.setWidget(1, QtGui.QFormLayout.LabelRole, self.label_2)
        self.costingArea = QtGui.QLineEdit(Dialog)
        self.costingArea.setObjectName(_fromUtf8("costingArea"))
        self.formLayout_3.setWidget(1, QtGui.QFormLayout.FieldRole, self.costingArea)
        self.label_7 = QtGui.QLabel(Dialog)
        self.label_7.setObjectName(_fromUtf8("label_7"))
        self.formLayout_3.setWidget(2, QtGui.QFormLayout.LabelRole, self.label_7)
        self.connectionType = QtGui.QLabel(Dialog)
        self.connectionType.setObjectName(_fromUtf8("connectionType"))
        self.formLayout_3.setWidget(0, QtGui.QFormLayout.FieldRole, self.connectionType)
        self.group = QtGui.QLineEdit(Dialog)
        self.group.setEnabled(False)
        self.group.setReadOnly(True)
        self.group.setObjectName(_fromUtf8("group"))
        self.formLayout_3.setWidget(2, QtGui.QFormLayout.FieldRole, self.group)
        self.verticalLayout.addLayout(self.formLayout_3)
        self.sometab = QtGui.QTabWidget(Dialog)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(9)
        sizePolicy.setHeightForWidth(self.sometab.sizePolicy().hasHeightForWidth())
        self.sometab.setSizePolicy(sizePolicy)
        self.sometab.setObjectName(_fromUtf8("sometab"))
        self.tab = QtGui.QWidget()
        self.tab.setObjectName(_fromUtf8("tab"))
        self.gridLayout = QtGui.QGridLayout(self.tab)
        self.gridLayout.setObjectName(_fromUtf8("gridLayout"))
        self.label_3 = QtGui.QLabel(self.tab)
        self.label_3.setObjectName(_fromUtf8("label_3"))
        self.gridLayout.addWidget(self.label_3, 0, 0, 1, 1)
        self.strengthMean = QtGui.QLineEdit(self.tab)
        self.strengthMean.setObjectName(_fromUtf8("strengthMean"))
        self.gridLayout.addWidget(self.strengthMean, 0, 1, 1, 1)
        self.label_4 = QtGui.QLabel(self.tab)
        self.label_4.setObjectName(_fromUtf8("label_4"))
        self.gridLayout.addWidget(self.label_4, 0, 2, 1, 1)
        self.strengthSigma = QtGui.QLineEdit(self.tab)
        self.strengthSigma.setObjectName(_fromUtf8("strengthSigma"))
        self.gridLayout.addWidget(self.strengthSigma, 0, 3, 1, 1)
        self.plot_strength_button = QtGui.QPushButton(self.tab)
        self.plot_strength_button.setObjectName(_fromUtf8("plot_strength_button"))
        self.gridLayout.addWidget(self.plot_strength_button, 0, 4, 1, 1)
        self.plot_strength = MatplotlibWidget(self.tab)
        self.plot_strength.setObjectName(_fromUtf8("plot_strength"))
        self.gridLayout.addWidget(self.plot_strength, 1, 0, 1, 5)
        self.sometab.addTab(self.tab, _fromUtf8(""))
        self.tab_2 = QtGui.QWidget()
        self.tab_2.setObjectName(_fromUtf8("tab_2"))
        self.gridLayout_2 = QtGui.QGridLayout(self.tab_2)
        self.gridLayout_2.setObjectName(_fromUtf8("gridLayout_2"))
        self.label_5 = QtGui.QLabel(self.tab_2)
        self.label_5.setObjectName(_fromUtf8("label_5"))
        self.gridLayout_2.addWidget(self.label_5, 0, 0, 1, 1)
        self.deadloadMean = QtGui.QLineEdit(self.tab_2)
        self.deadloadMean.setObjectName(_fromUtf8("deadloadMean"))
        self.gridLayout_2.addWidget(self.deadloadMean, 0, 1, 1, 1)
        self.label_6 = QtGui.QLabel(self.tab_2)
        self.label_6.setObjectName(_fromUtf8("label_6"))
        self.gridLayout_2.addWidget(self.label_6, 0, 2, 1, 1)
        self.deadloadSigma = QtGui.QLineEdit(self.tab_2)
        self.deadloadSigma.setObjectName(_fromUtf8("deadloadSigma"))
        self.gridLayout_2.addWidget(self.deadloadSigma, 0, 3, 1, 1)
        self.plot_deadload_button = QtGui.QPushButton(self.tab_2)
        self.plot_deadload_button.setObjectName(_fromUtf8("plot_deadload_button"))
        self.gridLayout_2.addWidget(self.plot_deadload_button, 0, 4, 1, 1)
        self.plot_deadload = MatplotlibWidget(self.tab_2)
        self.plot_deadload.setObjectName(_fromUtf8("plot_deadload"))
        self.gridLayout_2.addWidget(self.plot_deadload, 1, 0, 1, 5)
        self.sometab.addTab(self.tab_2, _fromUtf8(""))
        self.tab_3 = QtGui.QWidget()
        self.tab_3.setObjectName(_fromUtf8("tab_3"))
        self.horizontalLayout_3 = QtGui.QHBoxLayout(self.tab_3)
        self.horizontalLayout_3.setObjectName(_fromUtf8("horizontalLayout_3"))
        self.connections = QtGui.QTableWidget(self.tab_3)
        self.connections.setProperty("showDropIndicator", False)
        self.connections.setDragDropOverwriteMode(False)
        self.connections.setAlternatingRowColors(False)
        self.connections.setSelectionMode(QtGui.QAbstractItemView.SingleSelection)
        self.connections.setSelectionBehavior(QtGui.QAbstractItemView.SelectRows)
        self.connections.setColumnCount(5)
        self.connections.setObjectName(_fromUtf8("connections"))
        self.connections.setRowCount(0)
        item = QtGui.QTableWidgetItem()
        self.connections.setHorizontalHeaderItem(0, item)
        item = QtGui.QTableWidgetItem()
        self.connections.setHorizontalHeaderItem(1, item)
        item = QtGui.QTableWidgetItem()
        self.connections.setHorizontalHeaderItem(2, item)
        item = QtGui.QTableWidgetItem()
        self.connections.setHorizontalHeaderItem(3, item)
        item = QtGui.QTableWidgetItem()
        self.connections.setHorizontalHeaderItem(4, item)
        self.horizontalLayout_3.addWidget(self.connections)
        self.sometab.addTab(self.tab_3, _fromUtf8(""))
        self.verticalLayout.addWidget(self.sometab)
        self.buttonBox = QtGui.QDialogButtonBox(Dialog)
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtGui.QDialogButtonBox.NoButton)
        self.buttonBox.setObjectName(_fromUtf8("buttonBox"))
        self.verticalLayout.addWidget(self.buttonBox)

        self.retranslateUi(Dialog)
        self.sometab.setCurrentIndex(0)
        QtCore.QObject.connect(self.buttonBox, QtCore.SIGNAL(_fromUtf8("accepted()")), Dialog.accept)
        QtCore.QObject.connect(self.buttonBox, QtCore.SIGNAL(_fromUtf8("rejected()")), Dialog.reject)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        Dialog.setWindowTitle(_translate("Dialog", "Connection Type Editor", None))
        self.label.setText(_translate("Dialog", "Connection Type:", None))
        self.label_2.setText(_translate("Dialog", "Costing Area:", None))
        self.label_7.setText(_translate("Dialog", "Group:", None))
        self.connectionType.setText(_translate("Dialog", "TextLabel", None))
        self.label_3.setText(_translate("Dialog", "Mean:", None))
        self.label_4.setText(_translate("Dialog", "Sigma:", None))
        self.plot_strength_button.setText(_translate("Dialog", "Plot", None))
        self.sometab.setTabText(self.sometab.indexOf(self.tab), _translate("Dialog", "Strength PDF", None))
        self.label_5.setText(_translate("Dialog", "Mean:", None))
        self.label_6.setText(_translate("Dialog", "Sigma:", None))
        self.plot_deadload_button.setText(_translate("Dialog", "Plot", None))
        self.sometab.setTabText(self.sometab.indexOf(self.tab_2), _translate("Dialog", "Deadload PDF", None))
        item = self.connections.horizontalHeaderItem(0)
        item.setText(_translate("Dialog", "Connection Type", None))
        item = self.connections.horizontalHeaderItem(1)
        item.setText(_translate("Dialog", "Zone", None))
        item = self.connections.horizontalHeaderItem(2)
        item.setText(_translate("Dialog", "Edge", None))
        item = self.connections.horizontalHeaderItem(3)
        item.setText(_translate("Dialog", "Average", None))
        item = self.connections.horizontalHeaderItem(4)
        item.setText(_translate("Dialog", "Count", None))
        self.sometab.setTabText(self.sometab.indexOf(self.tab_3), _translate("Dialog", "Connections", None))

from matplotlibwidget import MatplotlibWidget
import windsim_rc
