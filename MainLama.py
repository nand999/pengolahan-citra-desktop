from PyQt5 import QtCore, QtGui, QtWidgets,uic
from PyQt5.QtWidgets import QFileDialog, QGraphicsScene, QSlider, QDialog, QVBoxLayout, QLabel, QPushButton, QApplication, QGraphicsPixmapItem, QInputDialog

from PyQt5.QtGui import QPixmap, QImage, QColor, qRgb
import matplotlib.pyplot as plt
import tempfile
import os
import cv2
from PIL import Image
import numpy as np
from histogram_rgb import HistogramDialog
from cropdialog import CropDialog
from tentang import tentangMain
from contrast import ContrastDialog

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1163, 623)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.graphicsView = QtWidgets.QGraphicsView(self.centralwidget)
        self.graphicsView.setGeometry(QtCore.QRect(30, 30, 521, 401))
        self.graphicsView.setObjectName("graphicsView")
        self.graphicsView_2 = QtWidgets.QGraphicsView(self.centralwidget)
        self.graphicsView_2.setGeometry(QtCore.QRect(610, 30, 521, 401))
        self.graphicsView_2.setObjectName("graphicsView_2")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1163, 26))
        self.menubar.setObjectName("menubar")
        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        self.menuINput = QtWidgets.QMenu(self.menubar)
        self.menuINput.setObjectName("menuINput")
        self.menuHIstogram = QtWidgets.QMenu(self.menuINput)
        self.menuHIstogram.setObjectName("menuHIstogram")
        self.menuColors = QtWidgets.QMenu(self.menubar)
        self.menuColors.setObjectName("menuColors")
        self.menuRGB = QtWidgets.QMenu(self.menuColors)
        self.menuRGB.setObjectName("menuRGB")
        self.menuRGB_to_Grayscale = QtWidgets.QMenu(self.menuColors)
        self.menuRGB_to_Grayscale.setObjectName("menuRGB_to_Grayscale")
        self.menuBit_Depth = QtWidgets.QMenu(self.menuColors)
        self.menuBit_Depth.setObjectName("menuBit_Depth")
        
        self.menuTentang = QtWidgets.QMenu(self.menubar)
        self.menuTentang.setObjectName("menuTentang")
        self.menuTentang.aboutToShow.connect(self.show_tentang_dialog)

        self.menuHistogram_Equalization = QtWidgets.QMenu(self.menubar)
        self.menuHistogram_Equalization.setObjectName("menuHistogram_Equalization")
        self.menuHistogram_Equalization.triggered.connect(self.histogram_equalization)

        self.menuAritmetical_Operation = QtWidgets.QMenu(self.menubar)
        self.menuAritmetical_Operation.setObjectName("menuAritmetical_Operation")

        self.menuFilter = QtWidgets.QMenu(self.menubar)
        self.menuFilter.setObjectName("menuFilter")
        self.menuEdge_Detection = QtWidgets.QMenu(self.menuFilter)
        self.menuEdge_Detection.setObjectName("menuEdge_Detection")
        self.menuGaussian_Blur = QtWidgets.QMenu(self.menuFilter)
        self.menuGaussian_Blur.setObjectName("menuGaussian_Blur")
        self.menuEdge_Detection_2 = QtWidgets.QMenu(self.menubar)
        self.menuEdge_Detection_2.setObjectName("menuEdge_Detection_2")
        self.menuMorfologi = QtWidgets.QMenu(self.menubar)
        self.menuMorfologi.setObjectName("menuMorfologi")
        self.menuErosion = QtWidgets.QMenu(self.menuMorfologi)
        self.menuErosion.setObjectName("menuErosion")
        self.menuDilation = QtWidgets.QMenu(self.menuMorfologi)
        self.menuDilation.setObjectName("menuDilation")
        self.menuOpening = QtWidgets.QMenu(self.menuMorfologi)
        self.menuOpening.setObjectName("menuOpening")
        self.menuClosing = QtWidgets.QMenu(self.menuMorfologi)
        self.menuClosing.setObjectName("menuClosing")

        self.menuClear = QtWidgets.QMenu(self.menubar)
        self.menuClear.setObjectName("menuClear")
        self.menuClear.aboutToShow.connect(self.clear_scene)

        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        
        MainWindow.setStatusBar(self.statusbar)
        self.actionOpenFile = QtWidgets.QAction(MainWindow)
        
        # application open
        self.actionOpenFile.setIconVisibleInMenu(True)
        self.actionOpenFile.setObjectName("actionOpenFile")
        self.actionOpenFile.triggered.connect(self.openImage)

        # Initialize a scene for the GraphicsView
        self.scene = QGraphicsScene()
        self.graphicsView.setScene(self.scene)

        # Initialize a scene for the output
        self.sceneOutput = QGraphicsScene()
        self.graphicsView_2.setScene(self.sceneOutput)

        self.actionSaveAs = QtWidgets.QAction(MainWindow)
        self.actionSaveAs.setObjectName("actionSaveAs")
        self.actionSaveAs.triggered.connect(self.saveAs)
        
        # application quit
        self.actionKeluar = QtWidgets.QAction(MainWindow)
        self.actionKeluar.setObjectName("actionKeluar")
        self.actionKeluar.triggered.connect(QtWidgets.QApplication.quit)
        
        self.actionInput = QtWidgets.QAction(MainWindow)
        self.actionInput.setObjectName("actionInput")
        self.actionInput.triggered.connect(self.histogram_input)

        self.actionOutput = QtWidgets.QAction(MainWindow)
        self.actionOutput.setObjectName("actionOutput")
        self.actionOutput.triggered.connect(self.histogram_output)

        self.actionInput_Output = QtWidgets.QAction(MainWindow)
        self.actionInput_Output.setObjectName("actionInput_Output")
        self.actionInput_Output.triggered.connect(self.histogram_input_output)

        # Invers negative
        self.actionInvers = QtWidgets.QAction(MainWindow)
        self.actionInvers.setObjectName("actionInvers")
        self.actionInvers.triggered.connect(self.negative_inverse)
        
        # Log Brightness
        self.actionLog_Brightness = QtWidgets.QAction(MainWindow)
        self.actionLog_Brightness.setObjectName("actionLog_Brightness")
        self.actionLog_Brightness.triggered.connect(self.log_brightness)
        
        
        self.actionGamma_Correction = QtWidgets.QAction(MainWindow)
        self.actionGamma_Correction.setObjectName("actionGamma_Correction")
        self.actionGamma_Correction.triggered.connect(lambda: self.gamma_correction(gamma=2.0))

        self.actionKuning = QtWidgets.QAction(MainWindow)
        self.actionKuning.setObjectName("actionKuning")
        self.actionKuning.triggered.connect(self.filter_kuning)

        self.actionOrange = QtWidgets.QAction(MainWindow)
        self.actionOrange.setObjectName("actionOrange")
        self.actionOrange.triggered.connect(self.filter_orange)

        self.actionCyan = QtWidgets.QAction(MainWindow)
        self.actionCyan.setObjectName("actionCyan")
        self.actionCyan.triggered.connect(self.filter_cyan)

        self.actionPurple = QtWidgets.QAction(MainWindow)
        self.actionPurple.setObjectName("actionPurple")
        self.actionPurple.triggered.connect(self.filter_purple)

        self.actionGrey = QtWidgets.QAction(MainWindow)
        self.actionGrey.setObjectName("actionGrey")
        self.actionGrey.triggered.connect(self.filter_grey)

        self.actionCoklat = QtWidgets.QAction(MainWindow)
        self.actionCoklat.setObjectName("actionCoklat")
        self.actionCoklat.triggered.connect(self.filter_coklat)

        self.actionMerah = QtWidgets.QAction(MainWindow)
        self.actionMerah.setObjectName("actionMerah")
        self.actionMerah.triggered.connect(self.filter_merah)

        # RGB to Grayscale Average method
        self.actionAverage = QtWidgets.QAction(MainWindow)
        self.actionAverage.setObjectName("actionAverage")
        self.actionAverage.triggered.connect(self.average)

        # RGB to Grayscale Lightness method
        self.actionLightness = QtWidgets.QAction(MainWindow)
        self.actionLightness.setObjectName("actionLightness")
        self.actionLightness.triggered.connect(self.lightness)

        # RGB to Grayscale Luminance method
        self.actionLuminance = QtWidgets.QAction(MainWindow)
        self.actionLuminance.setObjectName("actionLuminance")
        self.actionLuminance.triggered.connect(self.linear_luminance)



        #action brihtness
        self.menuBrightness = QtWidgets.QMenu(self.menuColors)
        self.menuBrightness.setObjectName("menuBrightness")
        self.actionlinearBrightness = QtWidgets.QAction(MainWindow)
        self.actionlinearBrightness.setObjectName("actionLinearBrightness")
        self.actionlinearBrightness.triggered.connect(self.linear_brightness)
        self.actionlinearBrightness.triggered.connect(self.show_brightness_slider)
        self.actionConstrast = QtWidgets.QAction(MainWindow)
        self.actionConstrast.setObjectName("actionConstrast")
        self.actionConstrast.triggered.connect(self.contrast)
        self.actionConstrast.triggered.connect(self.show_contrast_slider)
        self.actionSaturation = QtWidgets.QAction(MainWindow)
        self.actionSaturation.setObjectName("actionSaturation")
        self.actionSaturation.triggered.connect(self.linear_saturation)
        self.actionSaturation.triggered.connect(self.show_saturation_slider)

        
        # action bit depth
        self.action1_bit = QtWidgets.QAction(MainWindow)
        self.action1_bit.setObjectName("action1_bit")
        self.action1_bit.triggered.connect(lambda: self.bit_depht(1))
        self.action2_bit = QtWidgets.QAction(MainWindow)
        self.action2_bit.setObjectName("action2_bit")
        self.action2_bit.triggered.connect(lambda: self.bit_depht(2))
        self.action3_bit = QtWidgets.QAction(MainWindow)
        self.action3_bit.setObjectName("action3_bit")
        self.action3_bit.triggered.connect(lambda: self.bit_depht(3))
        self.action4_bit = QtWidgets.QAction(MainWindow)
        self.action4_bit.setObjectName("action4_bit")
        self.action4_bit.triggered.connect(lambda: self.bit_depht(4))
        self.action5_bit = QtWidgets.QAction(MainWindow)
        self.action5_bit.setObjectName("action5_bit")
        self.action5_bit.triggered.connect(lambda: self.bit_depht(5))
        self.action6_bit = QtWidgets.QAction(MainWindow)
        self.action6_bit.setObjectName("action6_bit")
        self.action6_bit.triggered.connect(lambda: self.bit_depht(6))
        self.action7_bit = QtWidgets.QAction(MainWindow)
        self.action7_bit.setObjectName("action7_bit")
        self.action7_bit.triggered.connect(lambda: self.bit_depht(7))

        #menu image processing
        self.actionHistogram_Equalization = QtWidgets.QAction(MainWindow)
        self.actionHistogram_Equalization.setObjectName("actionHistogram_Equalization")
        self.actionFuzzy_HE_RGB = QtWidgets.QAction(MainWindow)
        self.actionFuzzy_HE_RGB.setObjectName("actionFuzzy_HE_RGB")
        self.actionFuzzy_HE_RGB.triggered.connect(self.fuzzy_rgb)
        self.actionFuzzy_Grayscale = QtWidgets.QAction(MainWindow)
        self.actionFuzzy_Grayscale.setObjectName("actionFuzzy_Grayscale")
        self.actionFuzzy_Grayscale.triggered.connect(self.fuzzy_grayscale)

        self.actionIdentity = QtWidgets.QAction(MainWindow)
        self.actionIdentity.setObjectName("actionIdentity")

        self.actionSharpen = QtWidgets.QAction(MainWindow)
        self.actionSharpen.setObjectName("actionSharpen")

        self.actionUnsharp_Masking = QtWidgets.QAction(MainWindow)
        self.actionUnsharp_Masking.setObjectName("actionUnsharp_Masking")

        self.actionAverage_Filter = QtWidgets.QAction(MainWindow)
        self.actionAverage_Filter.setObjectName("actionAverage_Filter")

        self.actionLow_Pass_Filter = QtWidgets.QAction(MainWindow)
        self.actionLow_Pass_Filter.setObjectName("actionLow_Pass_Filter")

        self.actionHight_Pass_Filter = QtWidgets.QAction(MainWindow)
        self.actionHight_Pass_Filter.setObjectName("actionHight_Pass_Filter")

        self.actionBandstop_Filter = QtWidgets.QAction(MainWindow)
        self.actionBandstop_Filter.setObjectName("actionBandstop_Filter")

        self.actionEdge_Detection_1 = QtWidgets.QAction(MainWindow)
        self.actionEdge_Detection_1.setObjectName("actionEdge_Detection_1")

        self.actionEdge_Detection_2 = QtWidgets.QAction(MainWindow)
        self.actionEdge_Detection_2.setObjectName("actionEdge_Detection_2")

        self.actionEdge_Detection_3 = QtWidgets.QAction(MainWindow)
        self.actionEdge_Detection_3.setObjectName("actionEdge_Detection_3")

        self.actionGaussian_Blur_3x3 = QtWidgets.QAction(MainWindow)
        self.actionGaussian_Blur_3x3.setObjectName("actionGaussian_Blur_3x3")

        self.actionGaussian_Blur_3x5 = QtWidgets.QAction(MainWindow)
        self.actionGaussian_Blur_3x5.setObjectName("actionGaussian_Blur_3x5")

        #Edge detection
        self.actionPrewitt = QtWidgets.QAction(MainWindow)
        self.actionPrewitt.setObjectName("actionPrewitt")
        self.actionPrewitt.triggered.connect(self.prewitt_edge_detection)
        self.actionSebel = QtWidgets.QAction(MainWindow)
        self.actionSebel.setObjectName("actionSebel")
        self.actionSebel.triggered.connect(self.sobel_edge_detection)

        #menu geometri
        self.menuGeometri = QtWidgets.QMenu(self.menubar)
        self.menuGeometri.setObjectName("menuGeometri")
        self.actionScalingUniform = QtWidgets.QAction(MainWindow)
        self.actionScalingUniform.setObjectName("actionScalingUniform")
        self.actionScalingUniform.triggered.connect(self.scalingUniform)
        self.actionScalingNonUniform = QtWidgets.QAction(MainWindow)
        self.actionScalingNonUniform.setObjectName("actionScalingNonUniform")
        self.actionScalingNonUniform.triggered.connect(self.scalingNonUniform)
        self.actionCropping = QtWidgets.QAction(MainWindow)
        self.actionCropping.setObjectName("actionCropping")
        self.actionCropping.triggered.connect(self.show_crop_dialog)
        self.menuFlipping = QtWidgets.QMenu(self.menuGeometri)
        self.menuFlipping.setObjectName("menuFlipping")
        self.actionFlippingHorizontal = QtWidgets.QAction(MainWindow)
        self.actionFlippingHorizontal.setObjectName("actionFlippingHorizontal")
        self.actionFlippingHorizontal.triggered.connect(self.flipHorizontal)
        self.actionFlippingVertical = QtWidgets.QAction(MainWindow)
        self.actionFlippingVertical.setObjectName("actionFlippingVertical")
        self.actionFlippingVertical.triggered.connect(self.flipVertical)
        self.actionTranslasi = QtWidgets.QAction(MainWindow)
        self.actionTranslasi.setObjectName("actionTranslasi")
        self.actionTranslasi.triggered.connect(self.translasi)
        self.actionRotasi = QtWidgets.QAction(MainWindow)
        self.actionRotasi.setObjectName("actionRotasi")
        self.actionRotasi.triggered.connect(self.rotasi)
       
        #erosion
        self.actionSquare_4 = QtWidgets.QAction(MainWindow)
        self.actionSquare_4.setObjectName("actionSquare_4")
        self.actionSquare_4.triggered.connect(lambda: self.erosion('square', 4))
        self.actionSquare_6 = QtWidgets.QAction(MainWindow)
        self.actionSquare_6.setObjectName("actionSquare_6")
        self.actionSquare_6.triggered.connect(lambda: self.erosion('square', 6))
        self.actionCross_4 = QtWidgets.QAction(MainWindow)
        self.actionCross_4.setObjectName("actionCross_4")
        self.actionCross_4.triggered.connect(lambda: self.erosion('cross', 4))
        #dilation
        self.actionSquare_7 = QtWidgets.QAction(MainWindow)
        self.actionSquare_7.setObjectName("actionSquare_7")
        self.actionSquare_7.triggered.connect(lambda: self.dilation('square', 7))
        self.actionSquare_8 = QtWidgets.QAction(MainWindow)
        self.actionSquare_8.setObjectName("actionSquare_8")
        self.actionSquare_8.triggered.connect(lambda: self.dilation('square', 8))
        self.actionCross_5 = QtWidgets.QAction(MainWindow)
        self.actionCross_5.setObjectName("actionCross_5")
        self.actionCross_5.triggered.connect(lambda: self.dilation('cross', 5))
        #opening
        self.actionSquare_9 = QtWidgets.QAction(MainWindow)
        self.actionSquare_9.setObjectName("actionSquare_9")
        self.actionSquare_9.triggered.connect(lambda: self.opening('square', 9))
        #closing
        self.actionSquare_10 = QtWidgets.QAction(MainWindow)
        self.actionSquare_10.setObjectName("actionSquare_10")
        self.actionSquare_10.triggered.connect(lambda: self.closing('square', 10))

        self.actionTes2 = QtWidgets.QAction(MainWindow)
        self.actionTes2.setObjectName("actionTes2")

        #menu morfologi
        # self.

    
        #todo menu / pembuatan menu
        self.menuFile.addAction(self.actionOpenFile)
        self.menuFile.addAction(self.actionSaveAs)
        self.menuFile.addAction(self.actionKeluar)
        self.menuHIstogram.addAction(self.actionInput)
        self.menuHIstogram.addAction(self.actionOutput)
        self.menuHIstogram.addAction(self.actionInput_Output)
        self.menuINput.addAction(self.menuHIstogram.menuAction())
        self.menuRGB.addAction(self.actionKuning)
        self.menuRGB.addAction(self.actionOrange)
        self.menuRGB.addAction(self.actionCyan)
        self.menuRGB.addAction(self.actionPurple)
        self.menuRGB.addAction(self.actionGrey)
        self.menuRGB.addAction(self.actionCoklat)
        self.menuRGB.addAction(self.actionMerah)
        self.menuRGB_to_Grayscale.addAction(self.actionAverage)
        self.menuRGB_to_Grayscale.addAction(self.actionLightness)
        self.menuRGB_to_Grayscale.addAction(self.actionLuminance)
        self.menuBrightness.addAction(self.actionConstrast)
        self.menuBrightness.addAction(self.actionSaturation)
        self.menuBrightness.addAction(self.actionlinearBrightness)
        self.menuBit_Depth.addAction(self.action1_bit)
        self.menuBit_Depth.addAction(self.action2_bit)
        self.menuBit_Depth.addAction(self.action3_bit)
        self.menuBit_Depth.addAction(self.action4_bit)
        self.menuBit_Depth.addAction(self.action5_bit)
        self.menuBit_Depth.addAction(self.action6_bit)
        self.menuBit_Depth.addAction(self.action7_bit)
        self.menuColors.addAction(self.menuRGB.menuAction())
        self.menuColors.addAction(self.menuRGB_to_Grayscale.menuAction())
        self.menuColors.addAction(self.menuBrightness.menuAction())
        self.menuColors.addAction(self.actionInvers)
        self.menuColors.addAction(self.actionLog_Brightness)
        self.menuColors.addAction(self.menuBit_Depth.menuAction())
        self.menuColors.addAction(self.actionGamma_Correction)
        self.menuHistogram_Equalization.addAction(self.actionHistogram_Equalization)
        self.menuHistogram_Equalization.addAction(self.actionFuzzy_HE_RGB)
        self.menuHistogram_Equalization.addAction(self.actionFuzzy_Grayscale)
        self.menuEdge_Detection.addAction(self.actionEdge_Detection_1)
        self.menuEdge_Detection.addAction(self.actionEdge_Detection_2)
        self.menuEdge_Detection.addAction(self.actionEdge_Detection_3)
        self.menuGaussian_Blur.addAction(self.actionGaussian_Blur_3x3)
        self.menuGaussian_Blur.addAction(self.actionGaussian_Blur_3x5)
        self.menuFilter.addAction(self.actionIdentity)
        self.menuFilter.addAction(self.menuEdge_Detection.menuAction())
        self.menuFilter.addAction(self.actionSharpen)
        self.menuFilter.addAction(self.menuGaussian_Blur.menuAction())
        self.menuFilter.addAction(self.actionUnsharp_Masking)
        self.menuFilter.addAction(self.actionAverage_Filter)
        self.menuFilter.addAction(self.actionLow_Pass_Filter)
        self.menuFilter.addAction(self.actionHight_Pass_Filter)
        self.menuFilter.addAction(self.actionBandstop_Filter)
        self.menuEdge_Detection_2.addAction(self.actionPrewitt)
        self.menuEdge_Detection_2.addAction(self.actionSebel)
        self.menuErosion.addAction(self.actionSquare_4)
        self.menuErosion.addAction(self.actionSquare_6)
        self.menuErosion.addAction(self.actionCross_4)
        self.menuDilation.addAction(self.actionSquare_7)
        self.menuDilation.addAction(self.actionSquare_8)
        self.menuDilation.addAction(self.actionCross_5)
        self.menuOpening.addAction(self.actionSquare_9)
        self.menuClosing.addAction(self.actionSquare_10)

        self.menuMorfologi.addAction(self.menuErosion.menuAction())
        self.menuMorfologi.addAction(self.menuDilation.menuAction())
        self.menuMorfologi.addAction(self.menuOpening.menuAction())
        self.menuMorfologi.addAction(self.menuClosing.menuAction())

        self.menubar.addAction(self.menuFile.menuAction())
        self.menubar.addAction(self.menuINput.menuAction())
        self.menubar.addAction(self.menuColors.menuAction())
        self.menubar.addAction(self.menuTentang.menuAction())
        self.menubar.addAction(self.menuHistogram_Equalization.menuAction())
        self.menubar.addAction(self.menuAritmetical_Operation.menuAction())
        self.menubar.addAction(self.menuFilter.menuAction())
        self.menubar.addAction(self.menuEdge_Detection_2.menuAction())
        self.menubar.addAction(self.menuMorfologi.menuAction())
        self.menubar.addAction(self.menuGeometri.menuAction())
        self.menuGeometri.addAction(self.menuFlipping.menuAction())
        self.menuFlipping.addAction(self.actionFlippingHorizontal)
        self.menuFlipping.addAction(self.actionFlippingVertical)
        self.menuGeometri.addAction(self.actionScalingUniform)
        self.menuGeometri.addAction(self.actionScalingNonUniform)
        self.menuGeometri.addAction(self.actionCropping)
        self.menuGeometri.addAction(self.actionTranslasi)
        self.menuGeometri.addAction(self.actionRotasi)
        self.menubar.addAction(self.menuClear.menuAction())
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    #untuk penamaan menu dalam ui
    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.menuFile.setTitle(_translate("MainWindow", "File"))
        self.menuINput.setTitle(_translate("MainWindow", "View"))
        self.menuHIstogram.setTitle(_translate("MainWindow", "HIstogram"))
        self.menuColors.setTitle(_translate("MainWindow", "Colors"))
        self.menuRGB.setTitle(_translate("MainWindow", "RGB"))
        self.menuRGB_to_Grayscale.setTitle(_translate("MainWindow", "RGB to Grayscale"))
        self.menuBrightness.setTitle(_translate("MainWindow", "Brightness"))
        self.menuBit_Depth.setTitle(_translate("MainWindow", "Bit Depth"))
        self.menuTentang.setTitle(_translate("MainWindow", "Tentang"))
        self.menuHistogram_Equalization.setTitle(_translate("MainWindow", "Image Processing"))
        self.menuAritmetical_Operation.setTitle(_translate("MainWindow", "Aritmetical Operation"))
        self.menuFilter.setTitle(_translate("MainWindow", "Filter"))
        self.menuEdge_Detection.setTitle(_translate("MainWindow", "Edge Detection"))
        self.menuGaussian_Blur.setTitle(_translate("MainWindow", "Gaussian Blur"))
        self.menuEdge_Detection_2.setTitle(_translate("MainWindow", "Edge Detection"))
        self.menuMorfologi.setTitle(_translate("MainWindow", "Morfologi"))
        self.menuErosion.setTitle(_translate("MainWindow", "Erosion"))
        self.menuDilation.setTitle(_translate("MainWindow", "Dilation"))
        self.menuOpening.setTitle(_translate("MainWindow", "Opening"))
        self.menuClosing.setTitle(_translate("MainWindow", "Closing"))
        self.menuGeometri.setTitle(_translate("MainWindow", "Geometri"))
        self.actionRotasi.setText(_translate("MainWindow", "Rotasi"))
        self.actionTranslasi.setText(_translate("MainWindow", "Translasi"))
        self.menuFlipping.setTitle(_translate("MainWindow", "Flipping"))
        self.actionScalingUniform.setText(_translate("MainWindow", "Scaling Uniform"))
        self.actionScalingNonUniform.setText(_translate("MainWindow", "Scaling Non Uniform"))
        self.actionCropping.setText(_translate("MainWindow", "Cropping"))
        self.actionFlippingHorizontal.setText( _translate("MainWindow", "Flipping Horizontal"))
        self.actionFlippingVertical.setText(_translate("MainWindow", "Flipping Vertical"))
        self.menuClear.setTitle(_translate("MainWindow", "clear"))
        self.actionOpenFile.setText(_translate("MainWindow", "Open"))
        self.actionSaveAs.setText(_translate("MainWindow", "Save As"))
        self.actionKeluar.setText(_translate("MainWindow", "Exit"))
        self.actionInput.setText(_translate("MainWindow", "Input"))
        self.actionOutput.setText(_translate("MainWindow", "Output"))
        self.actionInput_Output.setText(_translate("MainWindow", "Input Output"))
        self.actionInvers.setText(_translate("MainWindow", "Invers"))
        self.actionLog_Brightness.setText(_translate("MainWindow", "Log Brightness"))
        self.actionGamma_Correction.setText(_translate("MainWindow", "Gamma Correction"))
        self.actionKuning.setText(_translate("MainWindow", "Yellow"))
        self.actionOrange.setText(_translate("MainWindow", "Orange"))
        self.actionCyan.setText(_translate("MainWindow", "Cyan"))
        self.actionPurple.setText(_translate("MainWindow", "Purple"))
        self.actionGrey.setText(_translate("MainWindow", "Grey"))
        self.actionCoklat.setText(_translate("MainWindow", "Brown"))
        self.actionMerah.setText(_translate("MainWindow", "Red"))
        self.actionAverage.setText(_translate("MainWindow", "Average"))
        self.actionLightness.setText(_translate("MainWindow", "Lightness"))
        self.actionLuminance.setText(_translate("MainWindow", "Luminance"))
        self.actionConstrast.setText(_translate("MainWindow", "Constrast"))
        self.actionlinearBrightness.setText(_translate("MainWindow", "Linear Brightness"))
        self.actionSaturation.setText(_translate("MainWindow", "Saturation"))
        self.action1_bit.setText(_translate("MainWindow", "1 bit"))
        self.action2_bit.setText(_translate("MainWindow", "2 bit"))
        self.action3_bit.setText(_translate("MainWindow", "3 bit"))
        self.action4_bit.setText(_translate("MainWindow", "4 bit"))
        self.action5_bit.setText(_translate("MainWindow", "5 bit"))
        self.action6_bit.setText(_translate("MainWindow", "6 bit"))
        self.action7_bit.setText(_translate("MainWindow", "7 bit"))
        self.actionHistogram_Equalization.setText(_translate("MainWindow", "Histogram Equalization"))
        self.actionFuzzy_HE_RGB.setText(_translate("MainWindow", "Fuzzy HE RGB"))
        self.actionFuzzy_Grayscale.setText(_translate("MainWindow", "Fuzzy Grayscale"))
        self.actionIdentity.setText(_translate("MainWindow", "Identity"))
        self.actionSharpen.setText(_translate("MainWindow", "Sharpen"))
        self.actionUnsharp_Masking.setText(_translate("MainWindow", "Unsharp Masking"))
        self.actionAverage_Filter.setText(_translate("MainWindow", "Average Filter"))
        self.actionLow_Pass_Filter.setText(_translate("MainWindow", "Low Pass Filter"))
        self.actionHight_Pass_Filter.setText(_translate("MainWindow", "High Pass Filter"))
        self.actionBandstop_Filter.setText(_translate("MainWindow", "Bandstop Filter"))
        self.actionEdge_Detection_1.setText(_translate("MainWindow", "Edge Detection 1"))
        self.actionEdge_Detection_2.setText(_translate("MainWindow", "Edge Detection 2"))
        self.actionEdge_Detection_3.setText(_translate("MainWindow", "Edge Detection 3"))
        self.actionGaussian_Blur_3x3.setText(_translate("MainWindow", "Gaussian Blur 3x3"))
        self.actionGaussian_Blur_3x5.setText(_translate("MainWindow", "Gaussian Blur 3x5"))
        self.actionPrewitt.setText(_translate("MainWindow", "Prewitt"))
        self.actionSebel.setText(_translate("MainWindow", "Sobel"))

        self.actionSquare_4.setText(_translate("MainWindow", "Square 4"))
        self.actionSquare_6.setText(_translate("MainWindow", "Square 6"))
        self.actionCross_4.setText(_translate("MainWindow", "Cross 4"))
        self.actionSquare_7.setText(_translate("MainWindow", "Square 7"))
        self.actionSquare_8.setText(_translate("MainWindow", "Square 8"))
        self.actionCross_5.setText(_translate("MainWindow", "Cross 5"))
        self.actionSquare_9.setText(_translate("MainWindow", "Square 9"))
        self.actionSquare_10.setText(_translate("MainWindow", "Square 10"))
        self.actionTes2.setText(_translate("MainWindow", "tes2"))


    def __init__(self):
        self.imagePath = None
        self.image_pixmap = None  
        self.imagefile = None
        self.imageResult = None
        self.directory_input = None
        self.directory_output = None
        self.histogram_input_dialog = None  
        self.histogram_output_dialog = None
        self.contrast_dialog = None

   
    def calculate_cumulative_histogram(self, histogram):
        # Menghitung cumulative histogram
        return np.cumsum(histogram)

    def histogram_input(self):
    # Pastikan bahwa self.directory_input menunjuk ke file gambar yang valid
        if not hasattr(self, 'directory_input') or not os.path.isfile(self.directory_input):
            QtWidgets.QMessageBox.critical(None, "Error", "File tidak ditemukan atau path tidak valid.")
            return

        # Cek apakah file dapat dibaca oleh OpenCV
        image = cv2.imread(self.directory_input)
        if image is None:
            QtWidgets.QMessageBox.critical(None, "Error", "Tidak dapat membaca file gambar. Periksa path dan format file.")
            return

        # Jika berhasil, lanjutkan membuka dialog histogram
        self.histogram_input_dialog = HistogramDialog(self.directory_input, 'Histogram Input')
        self.histogram_input_dialog.show()

    def histogram_output(self):
        # Pastikan directory input telah di-set
        if hasattr(self, 'directory_input'):
            output_file = "output.png"  # Nama file output yang akan digunakan

            # Cek apakah self.imageResult ada dan merupakan gambar
            if hasattr(self, 'imageResult'):
                # Gunakan gambar yang ada di self.imageResult (objek PIL.Image)
                output_image = self.imageResult

                # Simpan gambar ke file sementara
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
                    temp_file_path = temp_file.name
                    output_image.save(temp_file_path)

                # Muat gambar dari file sementara ke QPixmap
                img_pixmap = QtGui.QPixmap(temp_file_path)

                # Mendapatkan ukuran QGraphicsView
                view_width = self.graphicsView_2.width()
                view_height = self.graphicsView_2.height()

                # Skala pixmap agar sesuai dengan QGraphicsView, menjaga aspek rasio
                scaled_pixmap = img_pixmap.scaled(view_width, view_height, QtCore.Qt.KeepAspectRatio)

                # Bersihkan konten sebelumnya dan tambahkan pixmap baru ke scene
                self.sceneOutput.clear()
                self.sceneOutput.addPixmap(scaled_pixmap)
                self.graphicsView_2.setSceneRect(self.sceneOutput.itemsBoundingRect())

                # Tampilkan histogram output dengan dialog khusus
                self.histogram_output_dialog = HistogramDialog(
                    temp_file_path, 'Histogram Output')
                self.histogram_output_dialog.show()

                # Hapus file sementara
                os.remove(temp_file_path)
            else:
                QtWidgets.QMessageBox.critical(
                    None, "Error", "Tidak ada gambar yang dimuat.")
        else:
            QtWidgets.QMessageBox.critical(
                None, "Error", "Tidak ada gambar yang dimuat.")

    def histogram_input_output(self):
        self.histogram_input()
        self.histogram_output()

    def plot_histogram(self, red_histogram, green_histogram, blue_histogram, title):
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 3, 1)
        plt.bar(range(256), red_histogram, color='r', alpha=0.6)
        plt.xlabel('Pixel Value')
        plt.ylabel('Frequency')
        plt.title('Red Channel ' + title)

        plt.subplot(1, 3, 2)
        plt.bar(range(256), green_histogram, color='g', alpha=0.6)
        plt.xlabel('Pixel Value')
        plt.ylabel('Frequency')
        plt.title('Green Channel ' + title)

        plt.subplot(1, 3, 3)
        plt.bar(range(256), blue_histogram, color='b', alpha=0.6)
        plt.xlabel('Pixel Value')
        plt.ylabel('Frequency')
        plt.title('Blue Channel ' + title)

        plt.tight_layout()
        plt.show()

    def linear_brightness(self, brightness_factor):

        image = cv2.imread(self.imagePath, cv2.COLOR_BGR2RGB)

        # KOnversi ke numpy array
        image_np = np.array(image, dtype=np.int16)

        # Fungsi linear brightness
        def linear_brightness(image_np, brightness_factor):
            image_np = image_np + brightness_factor
            return np.clip(image_np, 0, 255).astype(np.uint8)

        # Konversi citra menjadi linear brightness dan value brightness
        linear_brightness_image = linear_brightness(image_np, brightness_factor)
        linear_brightness_image = cv2.cvtColor(linear_brightness_image, cv2.COLOR_BGR2RGB)

        output_image = Image.fromarray(linear_brightness_image)
        self.imageResult = output_image

        # Save image ke temporary file
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
            temp_file_path = temp_file.name
            output_image.save(temp_file_path)
        
        # Load image dari temp file ke QPixmap
        img_pixmap = QtGui.QPixmap(temp_file_path)

        # Get ukuran QGraphicsView
        view_width = self.graphicsView_2.width()
        view_height = self.graphicsView_2.height()

        # scale pixmap ke QGraphicView
        scaled_pixmap = img_pixmap.scaled(view_width, view_height, QtCore.Qt.KeepAspectRatio)

        self.sceneOutput.clear() #clear gambar yang ada di QGraphicview_2
        self.sceneOutput.addPixmap(scaled_pixmap)
        self.graphicsView_2.setSceneRect(self.sceneOutput.itemsBoundingRect())

        os.remove(temp_file_path)  
    #fuction saturation
    def linear_saturation(self, saturation_factor):

        image = cv2.imread(self.imagePath, cv2.COLOR_BGR2RGB)

        # KOnversi ke 5 channel
        image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        image_hsv = image_hsv.astype(np.float32)

        # Fungsi linear saturation
        def linear_saturation(image_hsv, saturation_factor):
            image_hsv[..., 1] = image_hsv[..., 1] * saturation_factor
            image_hsv[..., 1] = np.clip(image_hsv[..., 1], 0, 255)
            image_hsv = image_hsv.astype(np.uint8)
            return cv2.cvtColor(image_hsv, cv2.COLOR_HSV2RGB)

        # Konversi citra menjadi linear saturation dan valuae linear saturation
        linear_saturation_image = linear_saturation(image_hsv, saturation_factor)
        linear_saturation_image = cv2.cvtColor(linear_saturation_image, cv2.COLOR_BGR2RGB)

        output_image = Image.fromarray(linear_saturation_image)
        self.imageResult = output_image

        # Save image ke temporary file
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
            temp_file_path = temp_file.name
            output_image.save(temp_file_path)
        
        # Load image dari temp file ke QPixmap
        img_pixmap = QtGui.QPixmap(temp_file_path)

        # Get ukuran QGraphicsView
        view_width = self.graphicsView_2.width()
        view_height = self.graphicsView_2.height()

        # scale pixmap ke QGraphicView
        scaled_pixmap = img_pixmap.scaled(view_width, view_height, QtCore.Qt.KeepAspectRatio)

        self.sceneOutput.clear() #clear gambar yang ada di QGraphicview_2
        self.sceneOutput.addPixmap(scaled_pixmap)
        self.graphicsView_2.setSceneRect(self.sceneOutput.itemsBoundingRect())

        os.remove(temp_file_path)

    def show_saturation_slider(self):
        dialog = QDialog(self.centralwidget)
        dialog.setWindowTitle("Adjust saturation")

        layout = QVBoxLayout()

        slider = QSlider(QtCore.Qt.Horizontal)
        slider.setMinimum(0)
        slider.setMaximum(100)
        slider.setValue(50)
        slider.setTickPosition(QSlider.TicksBelow)
        slider.setTickInterval(10)
        layout.addWidget(QLabel("Adjust saturation"))
        layout.addWidget(slider)

        value_label = QLabel(f"Saturation: {slider.value()}")
        layout.addWidget(value_label)

        slider.valueChanged.connect(lambda: value_label.setText(f"Saturation : {slider.value()}"))

        ok_button = QPushButton("Ok")
        ok_button.clicked.connect(lambda: self.apply_brightness_change(slider.value(),dialog))
        layout.addWidget(ok_button)

        dialog.setLayout(layout)
        dialog.exec_()

    def linear_brightness(self, brightness_factor):

        image = cv2.imread(self.imagePath, cv2.COLOR_BGR2RGB)

        # KOnversi ke numpy array
        image_np = np.array(image, dtype=np.int16)

        # Fungsi linear brightness
        def linear_brightness(image_np, brightness_factor):
            image_np = image_np + brightness_factor
            return np.clip(image_np, 0, 255).astype(np.uint8)

        # Konversi citra menjadi linear brightness dan value brightness
        linear_brightness_image = linear_brightness(image_np, brightness_factor)
        linear_brightness_image = cv2.cvtColor(linear_brightness_image, cv2.COLOR_BGR2RGB)

        output_image = Image.fromarray(linear_brightness_image)
        self.imageResult = output_image

        # Save image ke temporary file
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
            temp_file_path = temp_file.name
            output_image.save(temp_file_path)
        
        # Load image dari temp file ke QPixmap
        img_pixmap = QtGui.QPixmap(temp_file_path)

        # Get ukuran QGraphicsView
        view_width = self.graphicsView_2.width()
        view_height = self.graphicsView_2.height()

        # scale pixmap ke QGraphicView
        scaled_pixmap = img_pixmap.scaled(view_width, view_height, QtCore.Qt.KeepAspectRatio)

        self.sceneOutput.clear() #clear gambar yang ada di QGraphicview_2
        self.sceneOutput.addPixmap(scaled_pixmap)
        self.graphicsView_2.setSceneRect(self.sceneOutput.itemsBoundingRect())

        os.remove(temp_file_path)  

    def show_brightness_slider(self):
        dialog = QDialog(self.centralwidget)
        dialog.setWindowTitle("Adjust Brightness")

        layout = QVBoxLayout()

        slider = QSlider(QtCore.Qt.Horizontal)
        slider.setMinimum(-100)
        slider.setMaximum(100)
        slider.setValue(0)
        slider.setTickPosition(QSlider.TicksBelow)
        slider.setTickInterval(10)
        layout.addWidget(QLabel("Adjust Brightness"))
        layout.addWidget(slider)

        value_label = QLabel(f"Brightness: {slider.value()}")
        layout.addWidget(value_label)

        slider.valueChanged.connect(lambda: value_label.setText(f"Brightness : {slider.value()}"))

        ok_button = QPushButton("Ok")
        ok_button.clicked.connect(lambda: self.apply_brightness_change(slider.value(),dialog))
        layout.addWidget(ok_button)

        dialog.setLayout(layout)
        dialog.exec_()

    def apply_brightness_change(self, brigthness_value, dialog):
        self.linear_brightness(brigthness_value)
        dialog.accept()
    #end fuction saturation

    #functionn Contras
    def contrast(self, contrast_factor):

        image = cv2.imread(self.imagePath, cv2.COLOR_BGR2RGB)

        # KOnversi ke numpy array
        image_np = np.array(image, dtype=np.float32)

        # Fungsi contrast
        def contrast(image_np, contrast_factor):
            image_np = image_np * contrast_factor
            return np.clip(image_np, 0, 255).astype(np.uint8)

        # Konversi citra menjadi contrast dan value contrast
        contrast_image = contrast(image_np, contrast_factor)
        contrast_image = cv2.cvtColor(contrast_image, cv2.COLOR_BGR2RGB)

        output_image = Image.fromarray(contrast_image)
        self.imageResult = output_image

        # Save image ke temporary file
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
            temp_file_path = temp_file.name
            output_image.save(temp_file_path)
        
        # Load image dari temp file ke QPixmap
        img_pixmap = QtGui.QPixmap(temp_file_path)

        # Get ukuran QGraphicsView
        view_width = self.graphicsView_2.width()
        view_height = self.graphicsView_2.height()

        # scale pixmap ke QGraphicView
        scaled_pixmap = img_pixmap.scaled(view_width, view_height, QtCore.Qt.KeepAspectRatio)

        self.sceneOutput.clear() #clear gambar yang ada di QGraphicview_2
        self.sceneOutput.addPixmap(scaled_pixmap)
        self.graphicsView_2.setSceneRect(self.sceneOutput.itemsBoundingRect())

        os.remove(temp_file_path)

    def show_contrast_slider(self):
        dialog = QDialog(self.centralwidget)
        dialog.setWindowTitle("Adjust Contrast")

        layout = QVBoxLayout()

        slider = QSlider(QtCore.Qt.Horizontal)
        slider.setMinimum(50)
        slider.setMaximum(300)
        slider.setValue(150)
        slider.setTickPosition(QSlider.TicksBelow)
        slider.setTickInterval(25)
        layout.addWidget(QLabel("Adjust Contrast"))
        layout.addWidget(slider)

        value_label = QLabel(f"Contrast: {slider.value()/100:.2f}")
        layout.addWidget(value_label)

        slider.valueChanged.connect(lambda: value_label.setText(f"Contrast : {slider.value()/100:.2f}"))

        ok_button = QPushButton("Ok")
        ok_button.clicked.connect(lambda: self.apply_contrast_change(slider.value()/100,dialog))
        layout.addWidget(ok_button)

        dialog.setLayout(layout)
        dialog.exec_()

    def apply_contrast_change(self, contrast_value, dialog):
        self.contrast(contrast_value)
        dialog.accept()  
    #end function contras

    def show_image(self):
        self.imagefile.show()

    def prewitt_edge_detection(self):
        # Konversi gambar ke grayscale
        image = self.imagefile.convert("L")  # Pastikan gambar dalam mode grayscale
        
        # Konversi gambar ke array NumPy
        image_np = np.array(image)
        
        # Definisi kernel Prewitt untuk sumbu X dan Y
        kernelx = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
        kernely = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
        
        # Aplikasi filter Prewitt
        edge_prewitt_x = cv2.filter2D(image_np, -1, kernelx)
        edge_prewitt_y = cv2.filter2D(image_np, -1, kernely)
        
        # Menghitung magnitudo gradien
        edges = np.sqrt(np.square(edge_prewitt_x) + np.square(edge_prewitt_y))
        
        # Normalisasi hasil deteksi tepi
        edges = np.uint8((edges / np.max(edges)) * 255)
        
        # Konversi array hasil ke gambar
        image_out = Image.fromarray(edges)
        self.imageResult = image_out
        
        # Simpan gambar ke temporary file
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
            temp_file_path = temp_file.name
            image_out.save(temp_file_path)

        # Load gambar dari file sementara ke QPixmap
        img_pixmap = QtGui.QPixmap(temp_file_path)

        # Mendapatkan ukuran QGraphicsView
        view_width = self.graphicsView_2.width()
        view_height = self.graphicsView_2.height()

        # Mengatur pixmap untuk skala agar sesuai dengan ukuran QGraphicsView dengan tetap menjaga rasio
        scaled_pixmap = img_pixmap.scaled(view_width, view_height, QtCore.Qt.KeepAspectRatio)

        self.sceneOutput.clear()  # Bersihkan konten sebelumnya di scene
        self.sceneOutput.addPixmap(scaled_pixmap)
        self.graphicsView_2.setSceneRect(self.sceneOutput.itemsBoundingRect())

        # Hapus file sementara
        os.remove(temp_file_path)

    def sobel_edge_detection(self):
        # Konversi gambar ke grayscale
        image = self.imagefile.convert("L")  # Pastikan gambar dalam mode grayscale
        
        # Konversi gambar ke array NumPy
        image_np = np.array(image)
        
        # Aplikasi filter Sobel pada sumbu X dan Y
        sobelx = cv2.Sobel(image_np, cv2.CV_64F, 1, 0, ksize=3)  # Sobel pada arah X
        sobely = cv2.Sobel(image_np, cv2.CV_64F, 0, 1, ksize=3)  # Sobel pada arah Y
        
        # Menghitung magnitudo gradien
        edges = np.sqrt(np.square(sobelx) + np.square(sobely))
        
        # Normalisasi hasil deteksi tepi
        edges = np.uint8((edges / np.max(edges)) * 255)
        
        # Konversi array hasil ke gambar
        image_out = Image.fromarray(edges)
        self.imageResult = image_out
        
        # Simpan gambar ke temporary file
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
            temp_file_path = temp_file.name
            image_out.save(temp_file_path)

        # Load gambar dari file sementara ke QPixmap
        img_pixmap = QtGui.QPixmap(temp_file_path)

        # Mendapatkan ukuran QGraphicsView
        view_width = self.graphicsView_2.width()
        view_height = self.graphicsView_2.height()

        # Mengatur pixmap untuk skala agar sesuai dengan ukuran QGraphicsView dengan tetap menjaga rasio
        scaled_pixmap = img_pixmap.scaled(view_width, view_height, QtCore.Qt.KeepAspectRatio)

        self.sceneOutput.clear()  # Bersihkan konten sebelumnya di scene
        self.sceneOutput.addPixmap(scaled_pixmap)
        self.graphicsView_2.setSceneRect(self.sceneOutput.itemsBoundingRect())

        # Hapus file sementara
        os.remove(temp_file_path)

    def clear_scene(self):
        self.scene.clear()
        self.sceneOutput.clear()

    def show_tentang_dialog(self):
        # Create a new dialog and load the 'Tentang.ui' file
       tentang_dialog = tentangMain()
       tentang_dialog.exec_() 

    def linear_luminance(self):
        image = self.imagefile
        image_np = np.array(image)

        def rgb_to_grayscale_luminance(image):
            return (0.2989 * image[:,:,0] + 0.5870 * image[:,:,1] + 0.1140 * image[:,:,2]).astype(np.uint8)

        grayscale_image = rgb_to_grayscale_luminance(image_np)

        output_image = Image.fromarray(grayscale_image)
        self.imageResult = output_image

        # Save the image to a temporary file
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
            temp_file_path = temp_file.name
            output_image.save(temp_file_path)

        img_pixmap = QtGui.QPixmap(temp_file_path)

        view_width = self.graphicsView_2.width()
        view_height = self.graphicsView_2.height()

        scaled_pixmap = img_pixmap.scaled(view_width, view_height, QtCore.Qt.KeepAspectRatio)

        self.sceneOutput.clear()  
        self.sceneOutput.addPixmap(scaled_pixmap)
      
        self.graphicsView_2.setSceneRect(self.sceneOutput.itemsBoundingRect())
        
        # delete temp file
        os.remove(temp_file_path)

    def average(self):
        image = self.imagefile
        image_np = np.array(image)
        def rgb_to_grayscale_average(image):
            return np.mean(image, axis=2).astype(np.uint8)
        grayscale_image = rgb_to_grayscale_average(image_np)
        output_image = Image.fromarray(grayscale_image)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
            temp_file_path = temp_file.name
            output_image.save(temp_file_path)
        img_pixmap = QtGui.QPixmap(temp_file_path)
        view_width = self.graphicsView_2.width()
        view_height = self.graphicsView_2.height()
        scaled_pixmap = img_pixmap.scaled(view_width, view_height, QtCore.Qt.KeepAspectRatio)
        self.sceneOutput.clear()
        self.sceneOutput.addPixmap(scaled_pixmap)
        self.graphicsView_2.setSceneRect(self.sceneOutput.itemsBoundingRect())
        os.remove(temp_file_path)

    def lightness(self):
    # Pastikan gambar sudah dimuat
        if hasattr(self, 'imagefile'):
            image = self.imagefile
            image_np = np.array(image)

            # Fungsi untuk mengubah RGB ke grayscale menggunakan metode lightness
            def rgb_to_grayscale_lightness(image_np):
                max_rgb = np.max(image_np, axis=2)
                min_rgb = np.min(image_np, axis=2)
                lightness = ((max_rgb + min_rgb) / 2).astype(np.uint8)
                return lightness

            # Konversi gambar ke grayscale menggunakan lightness method
            grayscale_image_np = rgb_to_grayscale_lightness(image_np)
            grayscale_image = Image.fromarray(grayscale_image_np)

            # Simpan gambar hasil grayscale ke self.imageResult agar bisa disimpan
            self.imageResult = grayscale_image

            # Simpan gambar ke file sementara untuk ditampilkan
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
                temp_file_path = temp_file.name
                grayscale_image.save(temp_file_path)

            # Muat gambar dari file sementara ke QPixmap
            img_pixmap = QtGui.QPixmap(temp_file_path)

            # Mendapatkan ukuran QGraphicsView
            view_width = self.graphicsView_2.width()
            view_height = self.graphicsView_2.height()

            # Skala pixmap agar sesuai dengan QGraphicsView, menjaga aspek rasio
            scaled_pixmap = img_pixmap.scaled(view_width, view_height, QtCore.Qt.KeepAspectRatio)

            # Bersihkan konten sebelumnya di scene dan tambahkan scaled_pixmap
            self.sceneOutput.clear()
            self.sceneOutput.addPixmap(scaled_pixmap)
            self.graphicsView_2.setSceneRect(self.sceneOutput.itemsBoundingRect())

            # Hapus file sementara
            os.remove(temp_file_path)
        else:
            QtWidgets.QMessageBox.warning(
                None, "Error", "Gambar belum dimuat.")

    

    def negative_inverse(self):
        # Pastikan gambar sudah dimuat
        if hasattr(self, 'imagefile'):
            image = self.imagefile

            # Konversi gambar ke numpy array
            image_np = np.array(image)

            # Terapkan efek negative inverse (255 - pixel value)
            image_np = 255 - image_np

            # Konversi kembali ke format PIL image setelah efek diterapkan
            image_out = Image.fromarray(image_np.astype(np.uint8))

            # Simpan hasil gambar ke self.imageResult agar bisa disimpan
            self.imageResult = image_out

            # Simpan gambar hasil ke file sementara untuk ditampilkan di GUI
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
                temp_file_path = temp_file.name
                image_out.save(temp_file_path)

            # Muat gambar dari file sementara ke QPixmap
            img_pixmap = QtGui.QPixmap(temp_file_path)

            # Mendapatkan ukuran QGraphicsView
            view_width = self.graphicsView_2.width()
            view_height = self.graphicsView_2.height()

            # Skala pixmap agar sesuai dengan QGraphicsView, menjaga aspek rasio
            scaled_pixmap = img_pixmap.scaled(view_width, view_height, QtCore.Qt.KeepAspectRatio)

            # Bersihkan konten sebelumnya dan tambahkan pixmap baru ke scene
            self.sceneOutput.clear()
            self.sceneOutput.addPixmap(scaled_pixmap)
            self.graphicsView_2.setSceneRect(self.sceneOutput.itemsBoundingRect())

            # Hapus file sementara setelah selesai
            os.remove(temp_file_path)
        else:
            QtWidgets.QMessageBox.warning(
                None, "Error", "Gambar belum dimuat.")

    def log_brightness(self):
    # Membaca gambar dari direktori sebagai grayscale
        image = cv2.imread(self.imagePath, cv2.IMREAD_GRAYSCALE)

        # Konstanta untuk transformasi logaritmik
        c = 255 / np.log(1 + np.max(image))

        # Melakukan transformasi logaritmik
        log_transformed = c * np.log(1 + image)

        # Normalisasi gambar log transformasi
        log_transformed = (log_transformed - np.min(log_transformed)) / (np.max(log_transformed) - np.min(log_transformed)) * 255
        log_transformed = np.array(log_transformed, dtype=np.uint8)

        # Mengonversi array ke citra
        output_image = Image.fromarray(log_transformed)

        # Simpan gambar log transformasi ke file sementara
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
            temp_file_path = temp_file.name
            output_image.save(temp_file_path)

        # Load gambar dari file sementara ke QPixmap
        img_pixmap = QtGui.QPixmap(temp_file_path)

        # Mendapatkan ukuran QGraphicsView
        view_width = self.graphicsView_2.width()
        view_height = self.graphicsView_2.height()

        # Mengatur pixmap untuk skala agar sesuai dengan ukuran QGraphicsView dengan tetap menjaga rasio
        scaled_pixmap = img_pixmap.scaled(view_width, view_height, QtCore.Qt.KeepAspectRatio)

        # Bersihkan konten sebelumnya di scene dan tambahkan scaled_pixmap
        self.sceneOutput.clear()
        self.sceneOutput.addPixmap(scaled_pixmap)
        self.graphicsView_2.setSceneRect(self.sceneOutput.itemsBoundingRect())

        # Hapus file sementara
        os.remove(temp_file_path)

    def bit_depht(self, bit):
    # Path gambar input
        image_path = self.imagePath
        num_colors = bit

        # Fungsi kuantisasi citra
        def kuantisasi_image(image_path, num_colors):
            image = Image.open(image_path)
            image = image.convert('RGB')

            img_array = np.array(image)

            min_val = img_array.min()
            max_val = img_array.max()

            step_size = (max_val - min_val) / num_colors

            # Melakukan kuantisasi pada array gambar
            kuantisasi_array = (img_array // step_size) * step_size

            # Mengonversi kembali array yang sudah dikuantisasi ke gambar
            kuantisasi_image = Image.fromarray(kuantisasi_array.astype('uint8'))

            return kuantisasi_image

        # Kuantisasi gambar
        result_image = kuantisasi_image(image_path, num_colors)

        # Simpan gambar kuantisasi ke file sementara
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
            temp_file_path = temp_file.name
            result_image.save(temp_file_path)

        # Load gambar dari file sementara ke QPixmap
        img_pixmap = QtGui.QPixmap(temp_file_path)

        # Mendapatkan ukuran QGraphicsView
        view_width = self.graphicsView_2.width()
        view_height = self.graphicsView_2.height()

        # Mengatur pixmap untuk skala agar sesuai dengan ukuran QGraphicsView dengan tetap menjaga rasio
        scaled_pixmap = img_pixmap.scaled(view_width, view_height, QtCore.Qt.KeepAspectRatio)

        # Bersihkan konten sebelumnya di scene dan tambahkan scaled_pixmap
        self.sceneOutput.clear()
        self.sceneOutput.addPixmap(scaled_pixmap)
        self.graphicsView_2.setSceneRect(self.sceneOutput.itemsBoundingRect())

        # Hapus file sementara
        os.remove(temp_file_path)

    def saveAs(self):
        # Periksa apakah ada gambar hasil (self.imageResult)
        if hasattr(self, 'imageResult') and self.imageResult is not None:
            # Buka dialog untuk memilih lokasi dan memberi nama file
            options = QFileDialog.Options()
            fileName, _ = QFileDialog.getSaveFileName(None, "Save Image As", "", "PNG Files (*.png);;JPEG Files (*.jpg);;BMP Files (*.bmp)", options=options)
            
            if fileName:
                # Simpan gambar ke lokasi yang dipilih dengan nama file baru
                self.imageResult.save(fileName)
                print(f"Gambar disimpan di: {fileName}")
            else:
                print("Penyimpanan dibatalkan.")
        else:
            print("Tidak ada gambar untuk disimpan.")

    def openImage(self):
        # Show file dialog to select an image file
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(None, "Open Image", "", "Image Files (*.png *.jpg *.bmp *.jpeg)", options=options)
        
        if fileName:
            self.imagePath = fileName
            self.directory_input = self.imagePath  # Tambahkan ini agar directory_input diisi
            img = Image.open(fileName)
            self.imagefile = img
            
            # Convert the image to QPixmap and display it
            self.image_pixmap = QtGui.QPixmap(fileName)
            view_width = self.graphicsView.width()
            view_height = self.graphicsView.height()
            scaled_pixmap = self.image_pixmap.scaled(view_width, view_height, QtCore.Qt.KeepAspectRatio)
            
            self.scene.clear()  # Clear any previous content in the scene
            self.scene.addPixmap(scaled_pixmap)
            self.graphicsView.setSceneRect(self.scene.itemsBoundingRect())
               
    def histogram_equalization(self):
        image = cv2.imread(self.imagePath, cv2.IMREAD_GRAYSCALE)

        equalized_image = cv2.equalizeHist(image)

        output_image = Image.fromarray(equalized_image)
        self.imageResult = output_image

        # Save image ke temporary file
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
            temp_file_path = temp_file.name
            output_image.save(temp_file_path)
        
        # Load image dari temp file ke QPixmap
        img_pixmap = QtGui.QPixmap(temp_file_path)

        # Get ukuran QGraphicsView
        view_width = self.graphicsView_2.width()
        view_height = self.graphicsView_2.height()

        # scale pixmap ke QGraphicView
        scaled_pixmap = img_pixmap.scaled(view_width, view_height, QtCore.Qt.KeepAspectRatio)

        self.sceneOutput.clear() #clear gambar yang ada di QGraphicview_2
        self.sceneOutput.addPixmap(scaled_pixmap)
        self.graphicsView_2.setSceneRect(self.sceneOutput.itemsBoundingRect())

        os.remove(temp_file_path)

    def gamma_correction(self, gamma=1.0):
        image = self.imagefile
        image_np = np.array(image)

        # Apply gamma correction
        gamma_corrected = np.array(255 * (image_np / 255) ** gamma, dtype='uint8')

        image_out = Image.fromarray(gamma_corrected)
        self.imageResult = image_out

        # Save the image to a temporary file
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
            temp_file_path = temp_file.name
            image_out.save(temp_file_path)

        # Load the image from the temporary file into QPixmap
        img_pixmap = QtGui.QPixmap(temp_file_path)

        # Get the size of the QGraphicsView
        view_width = self.graphicsView_2.width()
        view_height = self.graphicsView_2.height()

        # Scale the pixmap to fit the QGraphicsView, preserving the aspect ratio
        scaled_pixmap = img_pixmap.scaled(view_width, view_height, QtCore.Qt.KeepAspectRatio)

        self.sceneOutput.clear()  # Clear any previous content in the scene
        self.sceneOutput.addPixmap(scaled_pixmap)
        self.graphicsView_2.setSceneRect(self.sceneOutput.itemsBoundingRect())

        # delete temp file
        os.remove(temp_file_path)

    def fuzzy_grayscale(self):
        alpha = 1
        image = self.imagefile  # Pastikan ini adalah objek PIL.Image
        image_np = np.array(image.convert('RGB'))  # Konversi ke format RGB

        # Mendapatkan ukuran gambar
        height, width, _ = image_np.shape

        # Inisialisasi histogram untuk gambar grayscale
        grayscale_histogram = [0] * 256

        # Hitung histogram untuk gambar grayscale
        for y in range(height):
            for x in range(width):
                r, g, b = image_np[y, x]
                grayscale_value = int(0.299 * r + 0.587 * g + 0.114 * b)
                grayscale_histogram[grayscale_value] += 1

        # Hitung cumulative histogram untuk gambar grayscale
        cumulative_histogram = np.cumsum(grayscale_histogram)
        equalized_histogram = [0] * 256

        # Terapkan fuzzy histogram equalization pada gambar grayscale
        for y in range(height):
            for x in range(width):
                r, g, b = image_np[y, x]
                grayscale_value = int(0.299 * r + 0.587 * g + 0.114 * b)
                fuzzy_grayscale = int(alpha * cumulative_histogram[grayscale_value] + (1 - alpha) * grayscale_value)
                # Pastikan nilai fuzzy_grayscale berada dalam rentang 0-255
                fuzzy_grayscale = min(max(fuzzy_grayscale, 0), 255)
                image_np[y, x] = [fuzzy_grayscale, fuzzy_grayscale, fuzzy_grayscale]

                equalized_histogram[fuzzy_grayscale] += 1

        # Simpan gambar hasil ke file sementara
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
            temp_file_path = temp_file.name
            image_out = Image.fromarray(image_np.astype(np.uint8))
            image_out.save(temp_file_path)

        # Muat gambar dari file sementara ke QPixmap
        img_pixmap = QPixmap(temp_file_path)

        # Mendapatkan ukuran QGraphicsView
        view_width = self.graphicsView_2.width()
        view_height = self.graphicsView_2.height()

        # Skala pixmap agar sesuai dengan QGraphicsView, menjaga aspek rasio
        scaled_pixmap = img_pixmap.scaled(view_width, view_height, QtCore.Qt.KeepAspectRatio)

        # Bersihkan konten sebelumnya dan tambahkan pixmap baru ke scene
        self.sceneOutput.clear()
        self.sceneOutput.addPixmap(scaled_pixmap)
        self.graphicsView_2.setSceneRect(self.sceneOutput.itemsBoundingRect())

        # Hapus file sementara
        os.remove(temp_file_path)

        # Tampilkan histogram sebelum dan sesudah equalisasi
        # plt.figure(figsize=(12, 5))
        # plt.subplot(1, 2, 1)
        # plt.bar(range(256), grayscale_histogram, color='b', alpha=0.6, label='Before Equalization')
        # plt.xlabel('Pixel Value')
        # plt.ylabel('Frequency')
        # plt.legend()

        # plt.subplot(1, 2, 2)
        # plt.bar(range(256), equalized_histogram, color='r', alpha=0.6, label='After Equalization')
        # plt.xlabel('Pixel Value')
        # plt.ylabel('Frequency')
        # plt.legend()

        # plt.tight_layout()
        # plt.show()
    
    def fuzzy_rgb(self):
       
        alpha = 1  # Mengatur alpha menjadi 1
        image = self.imagefile  # Pastikan ini adalah objek PIL.Image
        image_np = np.array(image.convert('RGB'))  # Konversi ke format RGB

        # Mendapatkan ukuran gambar
        height, width, _ = image_np.shape

        # Inisialisasi histogram untuk setiap saluran
        red_histogram = [0] * 256
        green_histogram = [0] * 256
        blue_histogram = [0] * 256

        # Hitung histogram untuk setiap saluran
        for y in range(height):
            for x in range(width):
                r, g, b = image_np[y, x]
                red_histogram[r] += 1
                green_histogram[g] += 1
                blue_histogram[b] += 1

        # # Hitung cumulative histogram untuk setiap saluran
        # red_cumulative_histogram = self.calculate_cumulative_histogram(red_histogram)
        # green_cumulative_histogram = self.calculate_cumulative_histogram(green_histogram)
        # blue_cumulative_histogram = self.calculate_cumulative_histogram(blue_histogram)

        # Terapkan fuzzy histogram equalization pada setiap saluran
        # for y in range(height):
        #     for x in range(width):
        #         r, g, b = image_np[y, x]
                # fuzzy_r = int(alpha * red_cumulative_histogram[r] + (1 - alpha) * r)
                # fuzzy_g = int(alpha * green_cumulative_histogram[g] + (1 - alpha) * g)
                # fuzzy_b = int(alpha * blue_cumulative_histogram[b] + (1 - alpha) * b)
                # Pastikan nilai fuzzy_r, fuzzy_g, fuzzy_b berada dalam rentang 0-255
                # fuzzy_r = min(max(fuzzy_r, 0), 255)
                # fuzzy_g = min(max(fuzzy_g, 0), 255)
                # fuzzy_b = min(max(fuzzy_b, 0), 255)
                # image_np[y, x] = [fuzzy_r, fuzzy_g, fuzzy_b]

        # Simpan gambar hasil ke file sementara
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
            temp_file_path = temp_file.name
            image_out = Image.fromarray(image_np.astype(np.uint8))
            image_out.save(temp_file_path)

        # Muat gambar dari file sementara ke QPixmap
        img_pixmap = QPixmap(temp_file_path)

        # Mendapatkan ukuran QGraphicsView
        view_width = self.graphicsView_2.width()
        view_height = self.graphicsView_2.height()

        # Skala pixmap agar sesuai dengan QGraphicsView, menjaga aspek rasio
        scaled_pixmap = img_pixmap.scaled(view_width, view_height, QtCore.Qt.KeepAspectRatio)

        # Bersihkan konten sebelumnya dan tambahkan pixmap baru ke scene
        self.sceneOutput.clear()
        self.sceneOutput.addPixmap(scaled_pixmap)
        self.graphicsView_2.setSceneRect(self.sceneOutput.itemsBoundingRect())

        # Hapus file sementara
        os.remove(temp_file_path)

        # Tampilkan histogram sebelum dan sesudah equalisasi
        # self.plot_histogram(red_histogram, green_histogram, blue_histogram, "Before Equalization")
        # self.plot_histogram(
        #     red_cumulative_histogram, green_cumulative_histogram, blue_cumulative_histogram, "After Equalization"
        # )

        # Tampilkan histogram sebelum dan sesudah equalisasi
       
    #start menu geometri   
    def rotasi(self):
        # Pastikan imagefile sudah dimuat
        if hasattr(self, 'imagefile'):
            # Mengambil gambar dari self.imagefile
            image = self.imagefile

            # Meminta pengguna memasukkan sudut rotasi
            angle, ok = QtWidgets.QInputDialog.getInt(
                None, "Rotate Image", "Masukkan sudut rotasi (derajat):")

            if ok:
                # Rotasi gambar menggunakan PIL dengan sudut yang dimasukkan
                rotated_image = image.rotate(angle, expand=True)

                # Simpan hasil rotasi ke dalam self.imageResult untuk bisa disimpan nanti
                self.imageResult = rotated_image

                # Simpan gambar hasil ke file sementara untuk ditampilkan
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
                    temp_file_path = temp_file.name
                    rotated_image.save(temp_file_path)

                # Muat gambar dari file sementara ke QPixmap
                img_pixmap = QtGui.QPixmap(temp_file_path)

                # Mendapatkan ukuran QGraphicsView
                view_width = self.graphicsView_2.width()
                view_height = self.graphicsView_2.height()

                # Skala pixmap agar sesuai dengan QGraphicsView, menjaga aspek rasio
                scaled_pixmap = img_pixmap.scaled(view_width, view_height, QtCore.Qt.KeepAspectRatio)

                # Bersihkan konten sebelumnya dan tambahkan pixmap baru ke scene
                self.sceneOutput.clear()
                self.sceneOutput.addPixmap(scaled_pixmap)
                self.graphicsView_2.setSceneRect(self.sceneOutput.itemsBoundingRect())

                # Hapus file sementara setelah semuanya selesai
                os.remove(temp_file_path)
            else:
                QtWidgets.QMessageBox.warning(
                    None, "Error", "Masukkan sudut rotasi yang valid.")
        else:
            QtWidgets.QMessageBox.warning(
                None, "Error", "Gambar belum dimuat.")

    def flipHorizontal(self):
        # Pastikan gambar telah dimuat ke self.imagefile
        if hasattr(self, 'imagefile'):
            # Mengambil gambar dari self.imagefile
            image = self.imagefile

            # Melakukan flip horizontal pada gambar menggunakan PIL
            flipped_image = image.transpose(Image.FLIP_LEFT_RIGHT)

            # Simpan gambar hasil flip ke self.imageResult agar bisa disimpan dengan saveAs
            self.imageResult = flipped_image

            # Simpan gambar hasil ke file sementara
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
                temp_file_path = temp_file.name
                flipped_image.save(temp_file_path)

            # Muat gambar dari file sementara ke QPixmap
            img_pixmap = QPixmap(temp_file_path)

            # Mendapatkan ukuran QGraphicsView
            view_width = self.graphicsView_2.width()
            view_height = self.graphicsView_2.height()

            # Skala pixmap agar sesuai dengan QGraphicsView, menjaga aspek rasio
            scaled_pixmap = img_pixmap.scaled(view_width, view_height, QtCore.Qt.KeepAspectRatio)

            # Bersihkan konten sebelumnya dan tambahkan pixmap baru ke scene
            self.sceneOutput.clear()
            self.sceneOutput.addPixmap(scaled_pixmap)
            self.graphicsView_2.setSceneRect(self.sceneOutput.itemsBoundingRect())

            # Hapus file sementara
            os.remove(temp_file_path)
        else:
            QtWidgets.QMessageBox.warning(
                None, "Error", "Gambar belum dimuat.")

    def flipVertical(self):
        # Pastikan gambar telah dimuat ke self.imagefile
        if hasattr(self, 'imagefile'):
            # Mengambil gambar dari self.imagefile
            image = self.imagefile

            # Melakukan flip vertical pada gambar menggunakan PIL
            flipped_image = image.transpose(Image.FLIP_TOP_BOTTOM)

            # Simpan gambar hasil flip ke self.imageResult agar bisa disimpan dengan saveAs
            self.imageResult = flipped_image

            # Simpan gambar hasil ke file sementara
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
                temp_file_path = temp_file.name
                flipped_image.save(temp_file_path)

            # Muat gambar dari file sementara ke QPixmap
            img_pixmap = QPixmap(temp_file_path)

            # Mendapatkan ukuran QGraphicsView
            view_width = self.graphicsView_2.width()
            view_height = self.graphicsView_2.height()

            # Skala pixmap agar sesuai dengan QGraphicsView, menjaga aspek rasio
            scaled_pixmap = img_pixmap.scaled(view_width, view_height, QtCore.Qt.KeepAspectRatio)

            # Bersihkan konten sebelumnya dan tambahkan pixmap baru ke scene
            self.sceneOutput.clear()
            self.sceneOutput.addPixmap(scaled_pixmap)
            self.graphicsView_2.setSceneRect(self.sceneOutput.itemsBoundingRect())

            # Hapus file sementara
            os.remove(temp_file_path)
        else:
            QtWidgets.QMessageBox.warning(
                None, "Error", "Gambar belum dimuat.")

    def cropping(self):
        # Pastikan gambar telah dimuat ke self.imagefile
        if hasattr(self, 'imagefile'):
            # Mengambil gambar dari self.imagefile
            image = self.imagefile

            # Mendapatkan input dari pengguna untuk crop area
            x, ok_x = QtWidgets.QInputDialog.getInt(None, "Crop Image", "Masukkan koordinat X:")
            y, ok_y = QtWidgets.QInputDialog.getInt(None, "Crop Image", "Masukkan koordinat Y:")
            width, ok_width = QtWidgets.QInputDialog.getInt(None, "Crop Image", "Masukkan lebar:")
            height, ok_height = QtWidgets.QInputDialog.getInt(None, "Crop Image", "Masukkan tinggi:")

            if ok_x and ok_y and ok_width and ok_height:
                # Menggunakan PIL untuk crop gambar
                cropped_image = image.crop((x, y, x + width, y + height))

                # Simpan gambar hasil crop ke self.imageResult agar bisa disimpan dengan saveAs
                self.imageResult = cropped_image

                # Simpan gambar hasil ke file sementara untuk ditampilkan
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
                    temp_file_path = temp_file.name
                    cropped_image.save(temp_file_path)

                # Muat gambar dari file sementara ke QPixmap
                img_pixmap = QPixmap(temp_file_path)

                # Mendapatkan ukuran QGraphicsView
                view_width = self.graphicsView_2.width()
                view_height = self.graphicsView_2.height()

                # Skala pixmap agar sesuai dengan QGraphicsView, menjaga aspek rasio
                scaled_pixmap = img_pixmap.scaled(view_width, view_height, QtCore.Qt.KeepAspectRatio)

                # Bersihkan konten sebelumnya dan tambahkan pixmap baru ke scene
                self.sceneOutput.clear()
                self.sceneOutput.addPixmap(scaled_pixmap)
                self.graphicsView_2.setSceneRect(self.sceneOutput.itemsBoundingRect())

                # Hapus file sementara
                os.remove(temp_file_path)
            else:
                QtWidgets.QMessageBox.warning(
                    None, "Error", "Masukkan nilai yang valid untuk X, Y, lebar, dan tinggi.")
        else:
            QtWidgets.QMessageBox.warning(
                None, "Error", "Gambar belum dimuat.")

    def scalingUniform(self):
        # Pastikan gambar telah dimuat ke self.imagefile
        if hasattr(self, 'imagefile'):
            # Mengambil gambar dari self.imagefile
            image = self.imagefile

            # Meminta pengguna memasukkan skala
            scale_factor, ok = QtWidgets.QInputDialog.getDouble(
                None, "Uniform Scaling", "Masukkan skala:"
            )

            if ok and scale_factor > 0:
                # Menghitung ukuran baru berdasarkan faktor skala
                new_width = int(image.width * scale_factor)
                new_height = int(image.height * scale_factor)

                # Mengubah ukuran gambar menggunakan metode resize dari PIL
                scaled_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

                # Simpan hasil scaling ke file sementara
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
                    temp_file_path = temp_file.name
                    scaled_image.save(temp_file_path)

                # Muat gambar dari file sementara ke QPixmap
                img_pixmap = QPixmap(temp_file_path)

                # Mendapatkan ukuran QGraphicsView
                view_width = self.graphicsView_2.width()
                view_height = self.graphicsView_2.height()

                # Skala pixmap agar sesuai dengan QGraphicsView, menjaga aspek rasio
                scaled_pixmap = img_pixmap.scaled(view_width, view_height, QtCore.Qt.KeepAspectRatio)

                # Bersihkan konten sebelumnya dan tambahkan pixmap baru ke scene
                self.sceneOutput.clear()
                self.sceneOutput.addPixmap(scaled_pixmap)
                self.graphicsView_2.setSceneRect(self.sceneOutput.itemsBoundingRect())

                # Hapus file sementara
                os.remove(temp_file_path)
            else:
                QtWidgets.QMessageBox.warning(
                    None, "Error", "Masukkan bilangan positif yang valid."
                )
        else:
            QtWidgets.QMessageBox.warning(
                None, "Error", "Gambar belum dimuat."
            )

    def scalingNonUniform(self):
        # Pastikan gambar telah dimuat ke self.imagefile
        if hasattr(self, 'imagefile'):
            # Mengambil gambar dari self.imagefile
            image = self.imagefile

            # Meminta pengguna memasukkan skala untuk sumbu X dan Y
            scale_factor_x, ok_x = QtWidgets.QInputDialog.getDouble(
                None, "Non-Uniform Scaling", "Masukkan skala-X:")
            scale_factor_y, ok_y = QtWidgets.QInputDialog.getDouble(
                None, "Non-Uniform Scaling", "Masukkan skala-Y:")

            if ok_x and ok_y and scale_factor_x > 0 and scale_factor_y > 0:
                # Menghitung ukuran baru berdasarkan faktor skala
                new_width = int(image.width * scale_factor_x)
                new_height = int(image.height * scale_factor_y)

                # Mengubah ukuran gambar menggunakan metode resize dari PIL
                scaled_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

                # Simpan hasil scaling ke self.imageResult agar bisa digunakan untuk saveAs
                self.imageResult = scaled_image

                # Simpan gambar hasil scaling ke file sementara
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
                    temp_file_path = temp_file.name
                    scaled_image.save(temp_file_path)

                # Muat gambar dari file sementara ke QPixmap
                img_pixmap = QPixmap(temp_file_path)

                # Mendapatkan ukuran QGraphicsView
                view_width = self.graphicsView_2.width()
                view_height = self.graphicsView_2.height()

                # Skala pixmap agar sesuai dengan QGraphicsView, menjaga aspek rasio
                scaled_pixmap = img_pixmap.scaled(view_width, view_height, QtCore.Qt.KeepAspectRatio)

                # Bersihkan konten sebelumnya dan tambahkan pixmap baru ke scene
                self.sceneOutput.clear()
                self.sceneOutput.addPixmap(scaled_pixmap)
                self.graphicsView_2.setSceneRect(self.sceneOutput.itemsBoundingRect())

                # Hapus file sementara
                os.remove(temp_file_path)
            else:
                QtWidgets.QMessageBox.warning(
                    None, "Error", "Masukkan bilangan positif yang valid untuk skala-X dan skala-Y.")
        else:
            QtWidgets.QMessageBox.warning(
                None, "Error", "Gambar belum dimuat.") 
    
    def show_crop_dialog(self):
        if not self.imagefile:
            QtWidgets.QMessageBox.warning(self, "Warning", "No image loaded.")
            return
        
        # Show the crop dialog
        dialog = CropDialog(self.imagefile, MainWindow)
        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            crop_rect = dialog.get_crop_rect()
            self.crop_image(crop_rect)
    
    def translasi(self):
        # Pastikan gambar telah dimuat ke self.imagefile
        if hasattr(self, 'imagefile'):
            # Mengambil gambar dari self.imagefile
            image = self.imagefile
            image_np = np.array(image.convert('RGB'))  # Konversi ke format RGB

            # Meminta pengguna memasukkan nilai tx (geser horizontal)
            tx, ok_tx = QtWidgets.QInputDialog.getInt(
                None, "Translate Image", "Masukkan nilai tx (geser horizontal):")

            if ok_tx:
                # Meminta pengguna memasukkan nilai ty (geser vertikal)
                ty, ok_ty = QtWidgets.QInputDialog.getInt(
                    None, "Translate Image", "Masukkan nilai ty (geser vertikal):")

                if ok_ty:
                    height, width, _ = image_np.shape

                    # Membuat array gambar baru dengan latar belakang hitam
                    translated_image_np = np.zeros_like(image_np)

                    # Melakukan translasi gambar
                    for y in range(height):
                        for x in range(width):
                            x_translated = x + tx
                            y_translated = y + ty
                            # Memeriksa apakah koordinat baru berada dalam batas gambar
                            if 0 <= x_translated < width and 0 <= y_translated < height:
                                translated_image_np[y_translated, x_translated] = image_np[y, x]

                    # Konversi hasil translasi kembali ke Image
                    translated_image = Image.fromarray(translated_image_np)

                    # Simpan hasil translasi ke self.imageResult agar bisa digunakan untuk saveAs
                    self.imageResult = translated_image

                    # Simpan gambar hasil ke file sementara
                    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
                        temp_file_path = temp_file.name
                        translated_image.save(temp_file_path)

                    # Muat gambar dari file sementara ke QPixmap
                    img_pixmap = QPixmap(temp_file_path)

                    # Mendapatkan ukuran QGraphicsView
                    view_width = self.graphicsView_2.width()
                    view_height = self.graphicsView_2.height()

                    # Skala pixmap agar sesuai dengan QGraphicsView, menjaga aspek rasio
                    scaled_pixmap = img_pixmap.scaled(view_width, view_height, QtCore.Qt.KeepAspectRatio)

                    # Bersihkan konten sebelumnya dan tambahkan pixmap baru ke scene
                    self.sceneOutput.clear()
                    self.sceneOutput.addPixmap(scaled_pixmap)
                    self.graphicsView_2.setSceneRect(self.sceneOutput.itemsBoundingRect())

                    # Hapus file sementara
                    os.remove(temp_file_path)
                else:
                    QtWidgets.QMessageBox.warning(None, "Error", "Masukkan nilai ty yang valid.")
            else:
                QtWidgets.QMessageBox.warning(None, "Error", "Masukkan nilai tx yang valid.")
        else:
            QtWidgets.QMessageBox.warning(None, "Error", "Gambar belum dimuat.")


    def crop_image(self, rect):
        # Ensure an image is loaded
        if not self.imagefile:
            QtWidgets.QMessageBox.warning(self, "Warning", "No image loaded.")
            return
        
        # Crop the image with the provided coordinates
        left, top, right, bottom = map(int, [rect.left(), rect.top(), rect.right(), rect.bottom()])
        cropped_image = self.imagefile.crop((left, top, right, bottom))
        self.imageResult = cropped_image

        # Save the cropped image to a temporary file and load it into QPixmap
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
            temp_file_path = temp_file.name
            cropped_image.save(temp_file_path)
        
        img_pixmap = QtGui.QPixmap(temp_file_path)
        self.sceneOutput.clear()
        self.sceneOutput.addPixmap(img_pixmap)
        self.graphicsView_2.setScene(self.sceneOutput)
        self.graphicsView_2.fitInView(self.sceneOutput.sceneRect(), QtCore.Qt.KeepAspectRatio)
        
        os.remove(temp_file_path)
    #end menu geometri


    # rgb function
    def filter_kuning(self):
        image = self.imagefile
        image_np = np.array(image)

        # Apply yellow filter (increase red and green, remove blue)
        image_np[:, :, 2] = 0  # Remove blue component

        image_out = Image.fromarray(image_np.astype(np.uint8))
        self.imageResult = image_out

        # Save the image to a temporary file
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
            temp_file_path = temp_file.name
            image_out.save(temp_file_path)

        # Load the image from the temporary file into QPixmap
        img_pixmap = QtGui.QPixmap(temp_file_path)

        # Get the size of the QGraphicsView
        view_width = self.graphicsView_2.width()
        view_height = self.graphicsView_2.height()

        # Scale the pixmap to fit the QGraphicsView, preserving the aspect ratio
        scaled_pixmap = img_pixmap.scaled(view_width, view_height, QtCore.Qt.KeepAspectRatio)

        self.sceneOutput.clear()  # Clear any previous content in the scene
        self.sceneOutput.addPixmap(scaled_pixmap)
        self.graphicsView_2.setSceneRect(self.sceneOutput.itemsBoundingRect())

        # delete temp file
        os.remove(temp_file_path)

    def filter_orange(self):
        image = self.imagefile
        image_np = np.array(image)

        # Apply orange filter (reduce blue component)
        image_np[:, :, 2] = image_np[:, :, 2] // 2  # Reduce blue component

        image_out = Image.fromarray(image_np.astype(np.uint8))
        self.imageResult = image_out

        # Save the image to a temporary file
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
            temp_file_path = temp_file.name
            image_out.save(temp_file_path)

        # Load the image from the temporary file into QPixmap
        img_pixmap = QtGui.QPixmap(temp_file_path)

        # Get the size of the QGraphicsView
        view_width = self.graphicsView_2.width()
        view_height = self.graphicsView_2.height()

        # Scale the pixmap to fit the QGraphicsView, preserving the aspect ratio
        scaled_pixmap = img_pixmap.scaled(view_width, view_height, QtCore.Qt.KeepAspectRatio)

        self.sceneOutput.clear()  # Clear any previous content in the scene
        self.sceneOutput.addPixmap(scaled_pixmap)
        self.graphicsView_2.setSceneRect(self.sceneOutput.itemsBoundingRect())

        # delete temp file
        os.remove(temp_file_path)

    def filter_cyan(self):
        image = self.imagefile
        image_np = np.array(image)

        # Apply cyan filter (remove red component)
        image_np[:, :, 0] = 0  # Remove red component

        image_out = Image.fromarray(image_np.astype(np.uint8))
        self.imageResult = image_out

        # Save the image to a temporary file
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
            temp_file_path = temp_file.name
            image_out.save(temp_file_path)

        # Load the image from the temporary file into QPixmap
        img_pixmap = QtGui.QPixmap(temp_file_path)

        # Get the size of the QGraphicsView
        view_width = self.graphicsView_2.width()
        view_height = self.graphicsView_2.height()

        # Scale the pixmap to fit the QGraphicsView, preserving the aspect ratio
        scaled_pixmap = img_pixmap.scaled(view_width, view_height, QtCore.Qt.KeepAspectRatio)

        self.sceneOutput.clear()  # Clear any previous content in the scene
        self.sceneOutput.addPixmap(scaled_pixmap)
        self.graphicsView_2.setSceneRect(self.sceneOutput.itemsBoundingRect())

        # delete temp file
        os.remove(temp_file_path)

    def filter_purple(self):
        image = self.imagefile
        image_np = np.array(image)

        # Apply purple filter (remove green component)
        image_np[:, :, 1] = 0  # Remove green component

        image_out = Image.fromarray(image_np.astype(np.uint8))
        self.imageResult = image_out

        # Save the image to a temporary file
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
            temp_file_path = temp_file.name
            image_out.save(temp_file_path)

        # Load the image from the temporary file into QPixmap
        img_pixmap = QtGui.QPixmap(temp_file_path)

        # Get the size of the QGraphicsView
        view_width = self.graphicsView_2.width()
        view_height = self.graphicsView_2.height()

        # Scale the pixmap to fit the QGraphicsView, preserving the aspect ratio
        scaled_pixmap = img_pixmap.scaled(view_width, view_height, QtCore.Qt.KeepAspectRatio)

        self.sceneOutput.clear()  # Clear any previous content in the scene
        self.sceneOutput.addPixmap(scaled_pixmap)
        self.graphicsView_2.setSceneRect(self.sceneOutput.itemsBoundingRect())

        # delete temp file
        os.remove(temp_file_path)

    def filter_grey(self):
        image = self.imagefile
        image_np = np.array(image)

        # Convert to greyscale
        grey_image = np.mean(image_np, axis=2).astype(np.uint8)

        image_out = Image.fromarray(grey_image)
        self.imageResult = image_out

        # Save the image to a temporary file
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
            temp_file_path = temp_file.name
            image_out.save(temp_file_path)

        # Load the image from the temporary file into QPixmap
        img_pixmap = QtGui.QPixmap(temp_file_path)

        # Get the size of the QGraphicsView
        view_width = self.graphicsView_2.width()
        view_height = self.graphicsView_2.height()

        # Scale the pixmap to fit the QGraphicsView, preserving the aspect ratio
        scaled_pixmap = img_pixmap.scaled(view_width, view_height, QtCore.Qt.KeepAspectRatio)

        self.sceneOutput.clear()  # Clear any previous content in the scene
        self.sceneOutput.addPixmap(scaled_pixmap)
        self.graphicsView_2.setSceneRect(self.sceneOutput.itemsBoundingRect())

        # delete temp file
        os.remove(temp_file_path)

    def filter_coklat(self):
        image = self.imagefile
        image_np = np.array(image)

        # Apply brown filter (adjust red, green, blue components)
        brown_filter = image_np.copy()
        brown_filter[:, :, 0] = brown_filter[:, :, 0] // 2  # Reduce blue
        brown_filter[:, :, 1] = brown_filter[:, :, 1] // 3  # Reduce green

        image_out = Image.fromarray(brown_filter.astype(np.uint8))
        self.imageResult = image_out

        # Save the image to a temporary file
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
            temp_file_path = temp_file.name
            image_out.save(temp_file_path)

        # Load the image from the temporary file into QPixmap
        img_pixmap = QtGui.QPixmap(temp_file_path)

        # Get the size of the QGraphicsView
        view_width = self.graphicsView_2.width()
        view_height = self.graphicsView_2.height()

        # Scale the pixmap to fit the QGraphicsView, preserving the aspect ratio
        scaled_pixmap = img_pixmap.scaled(view_width, view_height, QtCore.Qt.KeepAspectRatio)

        self.sceneOutput.clear()  # Clear any previous content in the scene
        self.sceneOutput.addPixmap(scaled_pixmap)
        self.graphicsView_2.setSceneRect(self.sceneOutput.itemsBoundingRect())

        # delete temp file
        os.remove(temp_file_path)

    def filter_merah(self):
        image = self.imagefile
        image_np = np.array(image)

        # Apply red filter (remove green and blue components)
        image_np[:, :, 0] = 0  # Remove blue component
        image_np[:, :, 1] = 0  # Remove green component

        image_out = Image.fromarray(image_np.astype(np.uint8))
        self.imageResult = image_out

        # Save the image to a temporary file
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
            temp_file_path = temp_file.name
            image_out.save(temp_file_path)

        # Load the image from the temporary file into QPixmap
        img_pixmap = QtGui.QPixmap(temp_file_path)

        # Get the size of the QGraphicsView
        view_width = self.graphicsView_2.width()
        view_height = self.graphicsView_2.height()

        # Scale the pixmap to fit the QGraphicsView, preserving the aspect ratio
        scaled_pixmap = img_pixmap.scaled(view_width, view_height, QtCore.Qt.KeepAspectRatio)

        self.sceneOutput.clear()  # Clear any previous content in the scene
        self.sceneOutput.addPixmap(scaled_pixmap)
        self.graphicsView_2.setSceneRect(self.sceneOutput.itemsBoundingRect())

        # delete temp file
        os.remove(temp_file_path)
    
    def filter_kuning(self):
        image = self.imagefile
        image_np = np.array(image)

        # Apply yellow filter (increase red and green, remove blue)
        image_np[:, :, 2] = 0  # Remove blue component

        image_out = Image.fromarray(image_np.astype(np.uint8))
        self.imageResult = image_out

        # Save the image to a temporary file
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
            temp_file_path = temp_file.name
            image_out.save(temp_file_path)

        # Load the image from the temporary file into QPixmap
        img_pixmap = QtGui.QPixmap(temp_file_path)

        # Get the size of the QGraphicsView
        view_width = self.graphicsView_2.width()
        view_height = self.graphicsView_2.height()

        # Scale the pixmap to fit the QGraphicsView, preserving the aspect ratio
        scaled_pixmap = img_pixmap.scaled(view_width, view_height, QtCore.Qt.KeepAspectRatio)

        self.sceneOutput.clear()  # Clear any previous content in the scene
        self.sceneOutput.addPixmap(scaled_pixmap)
        self.graphicsView_2.setSceneRect(self.sceneOutput.itemsBoundingRect())

        # delete temp file
        os.remove(temp_file_path)

    def filter_orange(self):
        image = self.imagefile
        image_np = np.array(image)

        # Apply orange filter (reduce blue component)
        image_np[:, :, 2] = image_np[:, :, 2] // 2  # Reduce blue component

        image_out = Image.fromarray(image_np.astype(np.uint8))
        self.imageResult = image_out

        # Save the image to a temporary file
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
            temp_file_path = temp_file.name
            image_out.save(temp_file_path)

        # Load the image from the temporary file into QPixmap
        img_pixmap = QtGui.QPixmap(temp_file_path)

        # Get the size of the QGraphicsView
        view_width = self.graphicsView_2.width()
        view_height = self.graphicsView_2.height()

        # Scale the pixmap to fit the QGraphicsView, preserving the aspect ratio
        scaled_pixmap = img_pixmap.scaled(view_width, view_height, QtCore.Qt.KeepAspectRatio)

        self.sceneOutput.clear()  # Clear any previous content in the scene
        self.sceneOutput.addPixmap(scaled_pixmap)
        self.graphicsView_2.setSceneRect(self.sceneOutput.itemsBoundingRect())

        # delete temp file
        os.remove(temp_file_path)

    def filter_cyan(self):
        image = self.imagefile
        image_np = np.array(image)

        # Apply cyan filter (remove red component)
        image_np[:, :, 0] = 0  # Remove red component

        image_out = Image.fromarray(image_np.astype(np.uint8))
        self.imageResult = image_out

        # Save the image to a temporary file
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
            temp_file_path = temp_file.name
            image_out.save(temp_file_path)

        # Load the image from the temporary file into QPixmap
        img_pixmap = QtGui.QPixmap(temp_file_path)

        # Get the size of the QGraphicsView
        view_width = self.graphicsView_2.width()
        view_height = self.graphicsView_2.height()

        # Scale the pixmap to fit the QGraphicsView, preserving the aspect ratio
        scaled_pixmap = img_pixmap.scaled(view_width, view_height, QtCore.Qt.KeepAspectRatio)

        self.sceneOutput.clear()  # Clear any previous content in the scene
        self.sceneOutput.addPixmap(scaled_pixmap)
        self.graphicsView_2.setSceneRect(self.sceneOutput.itemsBoundingRect())

        # delete temp file
        os.remove(temp_file_path)

    def filter_purple(self):
        image = self.imagefile
        image_np = np.array(image)

        # Apply purple filter (remove green component)
        image_np[:, :, 1] = 0  # Remove green component

        image_out = Image.fromarray(image_np.astype(np.uint8))
        self.imageResult = image_out

        # Save the image to a temporary file
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
            temp_file_path = temp_file.name
            image_out.save(temp_file_path)

        # Load the image from the temporary file into QPixmap
        img_pixmap = QtGui.QPixmap(temp_file_path)

        # Get the size of the QGraphicsView
        view_width = self.graphicsView_2.width()
        view_height = self.graphicsView_2.height()

        # Scale the pixmap to fit the QGraphicsView, preserving the aspect ratio
        scaled_pixmap = img_pixmap.scaled(view_width, view_height, QtCore.Qt.KeepAspectRatio)

        self.sceneOutput.clear()  # Clear any previous content in the scene
        self.sceneOutput.addPixmap(scaled_pixmap)
        self.graphicsView_2.setSceneRect(self.sceneOutput.itemsBoundingRect())

        # delete temp file
        os.remove(temp_file_path)

    def filter_grey(self):
        image = self.imagefile
        image_np = np.array(image)

        # Convert to greyscale
        grey_image = np.mean(image_np, axis=2).astype(np.uint8)

        image_out = Image.fromarray(grey_image)
        self.imageResult = image_out

        # Save the image to a temporary file
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
            temp_file_path = temp_file.name
            image_out.save(temp_file_path)

        # Load the image from the temporary file into QPixmap
        img_pixmap = QtGui.QPixmap(temp_file_path)

        # Get the size of the QGraphicsView
        view_width = self.graphicsView_2.width()
        view_height = self.graphicsView_2.height()

        # Scale the pixmap to fit the QGraphicsView, preserving the aspect ratio
        scaled_pixmap = img_pixmap.scaled(view_width, view_height, QtCore.Qt.KeepAspectRatio)

        self.sceneOutput.clear()  # Clear any previous content in the scene
        self.sceneOutput.addPixmap(scaled_pixmap)
        self.graphicsView_2.setSceneRect(self.sceneOutput.itemsBoundingRect())

        # delete temp file
        os.remove(temp_file_path)

    def filter_coklat(self):
        image = self.imagefile
        image_np = np.array(image)

        # Apply brown filter (adjust red, green, blue components)
        brown_filter = image_np.copy()
        brown_filter[:, :, 0] = brown_filter[:, :, 0] // 2  # Reduce blue
        brown_filter[:, :, 1] = brown_filter[:, :, 1] // 3  # Reduce green

        image_out = Image.fromarray(brown_filter.astype(np.uint8))
        self.imageResult = image_out

        # Save the image to a temporary file
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
            temp_file_path = temp_file.name
            image_out.save(temp_file_path)

        # Load the image from the temporary file into QPixmap
        img_pixmap = QtGui.QPixmap(temp_file_path)

        # Get the size of the QGraphicsView
        view_width = self.graphicsView_2.width()
        view_height = self.graphicsView_2.height()

        # Scale the pixmap to fit the QGraphicsView, preserving the aspect ratio
        scaled_pixmap = img_pixmap.scaled(view_width, view_height, QtCore.Qt.KeepAspectRatio)

        self.sceneOutput.clear()  # Clear any previous content in the scene
        self.sceneOutput.addPixmap(scaled_pixmap)
        self.graphicsView_2.setSceneRect(self.sceneOutput.itemsBoundingRect())

        # delete temp file
        os.remove(temp_file_path)

    def filter_merah(self):
        image = self.imagefile
        image_np = np.array(image)

        # Apply red filter (remove green and blue components)
        image_np[:, :, 0] = 0  # Remove blue component
        image_np[:, :, 1] = 0  # Remove green component

        image_out = Image.fromarray(image_np.astype(np.uint8))
        self.imageResult = image_out

        # Save the image to a temporary file
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
            temp_file_path = temp_file.name
            image_out.save(temp_file_path)

        # Load the image from the temporary file into QPixmap
        img_pixmap = QtGui.QPixmap(temp_file_path)

        # Get the size of the QGraphicsView
        view_width = self.graphicsView_2.width()
        view_height = self.graphicsView_2.height()

        # Scale the pixmap to fit the QGraphicsView, preserving the aspect ratio
        scaled_pixmap = img_pixmap.scaled(view_width, view_height, QtCore.Qt.KeepAspectRatio)

        self.sceneOutput.clear()  # Clear any previous content in the scene
        self.sceneOutput.addPixmap(scaled_pixmap)
        self.graphicsView_2.setSceneRect(self.sceneOutput.itemsBoundingRect())

        # delete temp file
        os.remove(temp_file_path)

    # end rgb function
    
    def erosion(self, kernel_shape='square', kernel_size=4):
        # Konversi gambar ke grayscale
        image = self.imagefile.convert("L")
        image_np = np.array(image)

        # Tentukan bentuk kernel
        if kernel_shape == 'square':
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
        elif kernel_shape == 'cross':
            kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (kernel_size, kernel_size))

        # Terapkan operasi erosi
        eroded_image = cv2.erode(image_np, kernel, iterations=1)

        # Konversi array ke gambar
        image_out = Image.fromarray(eroded_image)
        self.imageResult = image_out

        # Proses yang sama untuk menampilkan gambar di QGraphicsView
        self._display_image(image_out)

    def dilation(self, kernel_shape='square', kernel_size=4):
        # Konversi gambar ke grayscale
        image = self.imagefile.convert("L")
        image_np = np.array(image)

        # Tentukan bentuk kernel
        if kernel_shape == 'square':
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
        elif kernel_shape == 'cross':
            kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (kernel_size, kernel_size))

        # Terapkan operasi dilasi
        dilated_image = cv2.dilate(image_np, kernel, iterations=1)

        # Konversi array ke gambar
        image_out = Image.fromarray(dilated_image)
        self.imageResult = image_out

        # Proses yang sama untuk menampilkan gambar di QGraphicsView
        self._display_image(image_out)

    def opening(self, kernel_shape='square', kernel_size=4):
        # Konversi gambar ke grayscale
        image = self.imagefile.convert("L")
        image_np = np.array(image)

        # Tentukan bentuk kernel
        if kernel_shape == 'square':
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
        elif kernel_shape == 'cross':
            kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (kernel_size, kernel_size))

        # Terapkan operasi opening (erosion diikuti dengan dilation)
        opened_image = cv2.morphologyEx(image_np, cv2.MORPH_OPEN, kernel)

        # Konversi array ke gambar
        image_out = Image.fromarray(opened_image)
        self.imageResult = image_out

        # Proses yang sama untuk menampilkan gambar di QGraphicsView
        self._display_image(image_out)

    def closing(self, kernel_shape='square', kernel_size=4):
        # Konversi gambar ke grayscale
        image = self.imagefile.convert("L")
        image_np = np.array(image)

        # Tentukan bentuk kernel
        if kernel_shape == 'square':
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
        elif kernel_shape == 'cross':
            kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (kernel_size, kernel_size))

        # Terapkan operasi closing (dilation diikuti dengan erosion)
        closed_image = cv2.morphologyEx(image_np, cv2.MORPH_CLOSE, kernel)

        # Konversi array ke gambar
        image_out = Image.fromarray(closed_image)
        self.imageResult = image_out

        # Proses yang sama untuk menampilkan gambar di QGraphicsView
        self._display_image(image_out)

    def _display_image(self, image_out):
        # Simpan gambar ke temporary file
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
            temp_file_path = temp_file.name
            image_out.save(temp_file_path)

        # Load image dari file sementara ke QPixmap
        img_pixmap = QtGui.QPixmap(temp_file_path)

        # Mendapatkan ukuran QGraphicsView
        view_width = self.graphicsView_2.width()
        view_height = self.graphicsView_2.height()

        # Mengatur pixmap untuk skala agar sesuai dengan ukuran QGraphicsView dengan tetap menjaga rasio
        scaled_pixmap = img_pixmap.scaled(view_width, view_height, QtCore.Qt.KeepAspectRatio)

        self.sceneOutput.clear()  # Bersihkan konten sebelumnya di scene
        self.sceneOutput.addPixmap(scaled_pixmap)
        self.graphicsView_2.setSceneRect(self.sceneOutput.itemsBoundingRect())

        # Hapus file sementara
        os.remove(temp_file_path)



   

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
