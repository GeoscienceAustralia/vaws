# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ui/connection.ui'
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
        Dialog.resize(723, 529)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(_fromUtf8(":/icons/images/editresize.png")), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        Dialog.setWindowIcon(icon)
        self.verticalLayout = QtGui.QVBoxLayout(Dialog)
        self.verticalLayout.setObjectName(_fromUtf8("verticalLayout"))
        self.formLayout_3 = QtGui.QFormLayout()
        self.formLayout_3.setFieldGrowthPolicy(QtGui.QFormLayout.AllNonFixedFieldsGrow)
        self.formLayout_3.setLabelAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.formLayout_3.setObjectName(_fromUtf8("formLayout_3"))
        self.label = QtGui.QLabel(Dialog)
        font = QtGui.QFont()
        font.setItalic(False)
        self.label.setFont(font)
        self.label.setObjectName(_fromUtf8("label"))
        self.formLayout_3.setWidget(0, QtGui.QFormLayout.LabelRole, self.label)
        self.name = QtGui.QLabel(Dialog)
        self.name.setEnabled(True)
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setItalic(False)
        font.setWeight(75)
        self.name.setFont(font)
        self.name.setObjectName(_fromUtf8("name"))
        self.formLayout_3.setWidget(0, QtGui.QFormLayout.FieldRole, self.name)
        self.label_2 = QtGui.QLabel(Dialog)
        font = QtGui.QFont()
        font.setItalic(False)
        self.label_2.setFont(font)
        self.label_2.setObjectName(_fromUtf8("label_2"))
        self.formLayout_3.setWidget(1, QtGui.QFormLayout.LabelRole, self.label_2)
        self.zone_location = QtGui.QLineEdit(Dialog)
        self.zone_location.setEnabled(False)
        self.zone_location.setReadOnly(True)
        self.zone_location.setObjectName(_fromUtf8("zone_location"))
        self.formLayout_3.setWidget(1, QtGui.QFormLayout.FieldRole, self.zone_location)
        self.label_7 = QtGui.QLabel(Dialog)
        font = QtGui.QFont()
        font.setItalic(False)
        self.label_7.setFont(font)
        self.label_7.setObjectName(_fromUtf8("label_7"))
        self.formLayout_3.setWidget(2, QtGui.QFormLayout.LabelRole, self.label_7)
        self.edge = QtGui.QLineEdit(Dialog)
        self.edge.setEnabled(False)
        self.edge.setReadOnly(True)
        self.edge.setObjectName(_fromUtf8("edge"))
        self.formLayout_3.setWidget(2, QtGui.QFormLayout.FieldRole, self.edge)
        self.label_9 = QtGui.QLabel(Dialog)
        font = QtGui.QFont()
        font.setItalic(False)
        self.label_9.setFont(font)
        self.label_9.setObjectName(_fromUtf8("label_9"))
        self.formLayout_3.setWidget(3, QtGui.QFormLayout.LabelRole, self.label_9)
        self.connection_type = QtGui.QLineEdit(Dialog)
        self.connection_type.setEnabled(False)
        self.connection_type.setReadOnly(True)
        self.connection_type.setObjectName(_fromUtf8("connection_type"))
        self.formLayout_3.setWidget(3, QtGui.QFormLayout.FieldRole, self.connection_type)
        self.label_10 = QtGui.QLabel(Dialog)
        font = QtGui.QFont()
        font.setItalic(False)
        self.label_10.setFont(font)
        self.label_10.setObjectName(_fromUtf8("label_10"))
        self.formLayout_3.setWidget(4, QtGui.QFormLayout.LabelRole, self.label_10)
        self.connection_group = QtGui.QLineEdit(Dialog)
        self.connection_group.setEnabled(False)
        self.connection_group.setReadOnly(True)
        self.connection_group.setObjectName(_fromUtf8("connection_group"))
        self.formLayout_3.setWidget(4, QtGui.QFormLayout.FieldRole, self.connection_group)
        self.verticalLayout.addLayout(self.formLayout_3)
        self.results = QtGui.QTabWidget(Dialog)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(9)
        sizePolicy.setHeightForWidth(self.results.sizePolicy().hasHeightForWidth())
        self.results.setSizePolicy(sizePolicy)
        self.results.setObjectName(_fromUtf8("results"))
        self.tab_3 = QtGui.QWidget()
        self.tab_3.setObjectName(_fromUtf8("tab_3"))
        self.horizontalLayout_3 = QtGui.QHBoxLayout(self.tab_3)
        self.horizontalLayout_3.setObjectName(_fromUtf8("horizontalLayout_3"))
        self.influences = QtGui.QTableWidget(self.tab_3)
        self.influences.setShowGrid(False)
        self.influences.setObjectName(_fromUtf8("influences"))
        self.influences.setColumnCount(2)
        self.influences.setRowCount(0)
        item = QtGui.QTableWidgetItem()
        self.influences.setHorizontalHeaderItem(0, item)
        item = QtGui.QTableWidgetItem()
        self.influences.setHorizontalHeaderItem(1, item)
        self.horizontalLayout_3.addWidget(self.influences)
        self.results.addTab(self.tab_3, _fromUtf8(""))
        self.tab_5 = QtGui.QWidget()
        self.tab_5.setObjectName(_fromUtf8("tab_5"))
        self.horizontalLayout = QtGui.QHBoxLayout(self.tab_5)
        self.horizontalLayout.setObjectName(_fromUtf8("horizontalLayout"))
        self.patches = QtGui.QTableWidget(self.tab_5)
        self.patches.setObjectName(_fromUtf8("patches"))
        self.patches.setColumnCount(3)
        self.patches.setRowCount(0)
        item = QtGui.QTableWidgetItem()
        self.patches.setHorizontalHeaderItem(0, item)
        item = QtGui.QTableWidgetItem()
        self.patches.setHorizontalHeaderItem(1, item)
        item = QtGui.QTableWidgetItem()
        self.patches.setHorizontalHeaderItem(2, item)
        self.horizontalLayout.addWidget(self.patches)
        self.results.addTab(self.tab_5, _fromUtf8(""))
        self.tab = QtGui.QWidget()
        self.tab.setObjectName(_fromUtf8("tab"))
        self.horizontalLayout_2 = QtGui.QHBoxLayout(self.tab)
        self.horizontalLayout_2.setObjectName(_fromUtf8("horizontalLayout_2"))
        self.connections = QtGui.QTreeWidget(self.tab)
        self.connections.setObjectName(_fromUtf8("connections"))
        self.horizontalLayout_2.addWidget(self.connections)
        self.results.addTab(self.tab, _fromUtf8(""))
        self.verticalLayout.addWidget(self.results)
        self.buttonBox = QtGui.QDialogButtonBox(Dialog)
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtGui.QDialogButtonBox.NoButton)
        self.buttonBox.setObjectName(_fromUtf8("buttonBox"))
        self.verticalLayout.addWidget(self.buttonBox)

        self.retranslateUi(Dialog)
        self.results.setCurrentIndex(0)
        QtCore.QObject.connect(self.buttonBox, QtCore.SIGNAL(_fromUtf8("accepted()")), Dialog.accept)
        QtCore.QObject.connect(self.buttonBox, QtCore.SIGNAL(_fromUtf8("rejected()")), Dialog.reject)
        QtCore.QMetaObject.connectSlotsByName(Dialog)
        Dialog.setTabOrder(self.zone_location, self.edge)
        Dialog.setTabOrder(self.edge, self.influences)
        Dialog.setTabOrder(self.influences, self.buttonBox)
        Dialog.setTabOrder(self.buttonBox, self.results)
        Dialog.setTabOrder(self.results, self.connection_group)
        Dialog.setTabOrder(self.connection_group, self.connection_type)

    def retranslateUi(self, Dialog):
        Dialog.setWindowTitle(_translate("Dialog", "Connection Viewer", None))
        self.label.setText(_translate("Dialog", "Connection:", None))
        self.name.setText(_translate("Dialog", "TextLabel", None))
        self.label_2.setText(_translate("Dialog", "Zone Location:", None))
        self.label_7.setText(_translate("Dialog", "Edge:", None))
        self.label_9.setText(_translate("Dialog", "Connection Type:", None))
        self.label_10.setText(_translate("Dialog", "Connection Group:", None))
        item = self.influences.horizontalHeaderItem(0)
        item.setText(_translate("Dialog", "Name", None))
        item = self.influences.horizontalHeaderItem(1)
        item.setText(_translate("Dialog", "Influence Coeff", None))
        self.results.setTabText(self.results.indexOf(self.tab_3), _translate("Dialog", "Zone Influences", None))
        item = self.patches.horizontalHeaderItem(0)
        item.setText(_translate("Dialog", "Patched Connection", None))
        item = self.patches.horizontalHeaderItem(1)
        item.setText(_translate("Dialog", "Zone", None))
        item = self.patches.horizontalHeaderItem(2)
        item.setText(_translate("Dialog", "Patch", None))
        self.results.setTabText(self.results.indexOf(self.tab_5), _translate("Dialog", "Influence Patches", None))
        self.connections.headerItem().setText(0, _translate("Dialog", "Name", None))
        self.connections.headerItem().setText(1, _translate("Dialog", "Broke At", None))
        self.connections.headerItem().setText(2, _translate("Dialog", "Strength", None))
        self.connections.headerItem().setText(3, _translate("Dialog", "Dead Load", None))
        self.connections.headerItem().setText(4, _translate("Dialog", "Damaged Load", None))
        self.results.setTabText(self.results.indexOf(self.tab), _translate("Dialog", "Simulation Results", None))

import windsim_rc