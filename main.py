import sys
import os
import time
import cv2
import csv

import board
import busio as io
import adafruit_mlx90614

import RPi.GPIO as GPIO

import numpy as np
import tensorflow as tf
from utils import label_map_util
from utils import visualization_utils as vis_util

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtWidgets import QFileDialog, QMessageBox, QDockWidget, QListWidget, QLabel
from PyQt5.QtGui import *
from GUI import Ui_MainWindow  # 匯入建立的GUI.ui


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"





class Thread(QThread):
    #id = pyqtSignal(str)
    temp = pyqtSignal(str)
    dist = pyqtSignal(float)
    def run(self):
        #print("請靠卡感應及貼近感測器")
        count = 0
        #if ID !="":
        #while (ID !="") and (str(count)=="0") :
            #count +=1
            #print(count)
            #with open("database.csv",encoding="Big5") as csvfile:
                #reader = csv.DictReader(csvfile)
                #for row in reader:
                    #if row["RFID"] == ID:
                        
                        #self.id.emit(str(row["Name"]))
                        
        GPIO.output(15, GPIO.HIGH)
        time.sleep(0.00001)
        GPIO.output(15, GPIO.LOW)
        while GPIO.input(14) == 0:
            start_time = time.time()
        while GPIO.input(14) == 1:
            end_time = time.time()
            etime = end_time - start_time
            dist = 17150 * etime
            if dist<5:
                dist0 = round(dist,2)
                self.dist.emit(dist0)
        
        i2c = io.I2C(board.SCL, board.SDA, frequency=100000)
        mlx = adafruit_mlx90614.MLX90614(i2c)
        t = mlx.object_temperature
        self.temp.emit(str(round(t,2)))
                    
                    
                        
                        

class mainwindow(QtWidgets.QMainWindow,Ui_MainWindow):  
    def log(self):
        with open('log.csv',encoding="Big5", mode='a') as logfile:
            log_write = csv.writer(logfile,delimiter=',',quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writelog = log_write.writerow([str(self.datetime.text()), str(self.StuID.text()), str(self.name.text()), self.temperature.text(), str(class_name)] )
        
        
    def clear_function(self):
            self.name.setText("")
            self.ID.setText("")
            self.StuID.setText("")
            self.temperature.setText("")
            self.distance.setText("")
            self.datetime.setText("")
            self.Info.setText("")
            self.maskdetectorlabel.setPixmap(QPixmap())
            #self.realtimelabel.setPixmap(QPixmap())
            self.DEF.stop()

    def __init__(self):
        super(mainwindow, self).__init__()
        self.setupUi(self)
        self.startBTN.clicked.connect(self.startBTN_function)  # one_pushButton是對應的objectName
        self.stopBTN.clicked.connect(self.stopBTN_function)
        self.cleancardidBTN.clicked.connect(self.cleancardidBTN_function)
        self.thread = Thread()
        #self.thread.id.connect(self.id)
        self.thread.temp.connect(self.temp)
        self.thread.dist.connect(self.dist)
        
        #self.stopBTN.clicked.connect(self.stopBTN_function)
        self.cap = cv2.VideoCapture(0) # 准备获取图像
        self.cap.set(3, 640) #set width
        self.cap.set(4, 480) #set height
        self.ABC = QTimer()
        self.DEF = QTimer()
        self.ABC.timeout.connect(self.startBTN_function)
        self.DEF.timeout.connect(self.clear_function)
        # 將點選事件與槽函式進行連線
        self.realtimelabel.setScaledContents(True)
          
        # setting main window geometry
        desktop_geometry = QtWidgets.QApplication.desktop()  # 獲取螢幕大小
        main_window_width = desktop_geometry.width()  # 螢幕的寬
        main_window_height = desktop_geometry.height()  # 螢幕的高
        rect = self.geometry()  # 獲取視窗介面大小
        window_width = rect.width()  # 視窗介面的寬
        window_height = rect.height()  # 視窗介面的高
        x = (main_window_width - window_width) // 2  # 計算視窗左上角點橫座標
        y = (main_window_height - window_height) // 2  # 計算視窗左上角點縱座標
        self.setGeometry(x, y, window_width, window_height)  # 設定視窗介面在螢幕上的位置
        # 無邊框以及背景透明一般不會在主視窗中用到，一般使用在子視窗中，例如在子視窗中顯示gif提示載入資訊等等
        #self.setWindowFlags(Qt.FramelessWindowHint)  # 無邊框
        #self.setAttribute(Qt.WA_TranslucentBackground)  # 背景透明
        
        # Name of the directory containing the object detection module we're using
        MODEL_NAME = 'inference_graph'

        # Grab path to current working directory
        CWD_PATH = os.getcwd()

        # Path to frozen detection graph .pb file, which contains the model that is used
        # for object detection.
        PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

        # Path to label map file
        self.PATH_TO_LABELS = os.path.join(CWD_PATH,'labelmap.pbtxt')

        IMAGE_NAME = 'temp.jpg'

        # Path to image
        self.PATH_TO_IMAGE = os.path.join(CWD_PATH,IMAGE_NAME)

        # Number of classes the object detector can identify
        self.NUM_CLASSES = 2

        # Load the label map.
        # Label maps map indices to category names, so that when our convolution
        # network predicts `5`, we know that this corresponds to `king`.
        # Here we use internal utility functions, but anything that returns a
        # dictionary mapping integers to appropriate string labels would be fine
        self.label_map = label_map_util.load_labelmap(self.PATH_TO_LABELS)
        self.categories = label_map_util.convert_label_map_to_categories(self.label_map, max_num_classes=self.NUM_CLASSES, use_display_name=True)
        self.category_index = label_map_util.create_category_index(self.categories)

        self.confident = 0.85
        # Load the Tensorflow model into memory.
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

            self.sess = tf.Session(graph=detection_graph)

          # Define input and output tensors (i.e. data) for the object detection classifier

          # Input tensor is the image
        self.image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

          # Output tensors are the detection boxes, scores, and classes
          # Each box represents a part of the image where a particular object was detected
        self.detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

          # Each score represents level of confidence for each of the objects.
          # The score is shown on the result image, together with the class label.
        self.detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

          # Number of objects detected
        self.num_detections = detection_graph.get_tensor_by_name('num_detections:0')
        
        
        TRIGGER_PIN = 15
        ECHO_PIN = 14
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(TRIGGER_PIN, GPIO.OUT)
        GPIO.setup(ECHO_PIN, GPIO.IN)
        GPIO.setup(12, GPIO.OUT)
        GPIO.output(TRIGGER_PIN, GPIO.LOW)
        time.sleep(1)
        
        
    
    def startBTN_function(self):   # pushbutton對應的響應函式
        
        rval,frame = self.cap.read()
        self.frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = QtGui.QImage(self.frame, self.frame.shape[1], self.frame.shape[0], QtGui.QImage.Format_RGB888)
        self.realtimelabel.setPixmap(QPixmap.fromImage(image))
        self.realtimelabel.setScaledContents(True)
        data = QDateTime.currentDateTime()
        currTime = data.toString("yyyy-MM-dd hh:mm:ss")
        self.datetime.setText(str(currTime))
        global ID
        ID = self.ID.text()
        with open("database.csv",encoding="Big5") as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    
                    if row["RFID"] == ID  and ID !="":
                        self.name.setText(row["Name"])
                        self.StuID.setText(row["StuID"])
                        self.Info.setText("查詢中，請稍後")
                        ID = self.ID.text()
                        letgo =True
                        self.thread.start()
                        
                    if ID == "":
                        self.Info.setText("請貼近感測器及靠卡輸入卡號")
                        
                    if row["RFID"] != ID and ID !="":
                        #self.Info.setText("查詢中，請稍後")
                        self.Info.setText("查無資料，請先建檔案")
                        
        #self.stopBTN_function()
        
        
        #print(ID)
        
        self.ABC.start(1000./15)
    
    def stopBTN_function(self):   # pushbutton對應的響應函式
        self.ABC.stop()
        self.name.setText("")
        self.ID.setText("")
        self.StuID.setText("")
        self.temperature.setText("")
        self.distance.setText("")
        self.datetime.setText("")
        self.Info.setText("")
        self.maskdetectorlabel.setPixmap(QPixmap())
        self.realtimelabel.setPixmap(QPixmap())
               
    def cleancardidBTN_function(self):
        self.ID.setText("")
        
    
    def id(self, data):
        self.name.setText(data)
        
    def temp(self,data):
        self.temperature.setText(data)
        self.log()
        
    def dist(self, data):
        if (data <=4) and (self.ID.text()!="") :
            self.distance.setText(str(data))
            self.ID.setText("")
            self.DEF.start(10000)
            cv2.imwrite("temp.jpg", self.frame)
            image00 = cv2.imread(self.PATH_TO_IMAGE)
            image_expanded = np.expand_dims(image00, axis=0)
            
            # Perform the actual detection by running the model with the image as input
            (boxes, scores, classes, num) = self.sess.run(
                [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
                feed_dict={self.image_tensor: image_expanded})
            # Draw the results of the detection (aka 'visulaize the results')
            s_classes = classes[scores > self.confident]
            for i in range(len(s_classes)):
                if s_classes[i] in self.category_index.keys():
                    global class_name
                    class_name = self.category_index[s_classes[i]]['name']  # 得到英文class名稱
                    
                    if str(class_name) == 'good':
                              #print('warnning')
                              p = GPIO.PWM(12, 1)
                              p.start(0.01)
                              p.ChangeFrequency(1046)
                              time.sleep(0.15)
                              p.ChangeFrequency(1175)
                              time.sleep(0.15)
                              p.ChangeFrequency(1318)
                              time.sleep(0.15)
                              p.stop()
                              
                    if str(class_name) == 'bad':
                              #print('warnning')
                              p = GPIO.PWM(12, 1)
                              p.start(0.01)
                              p.ChangeFrequency(1318)
                              time.sleep(0.15)
                              p.ChangeFrequency(1175)
                              time.sleep(0.15)
                              p.ChangeFrequency(1046)
                              time.sleep(0.15)
                              p.stop()
                              
                              
                              
                              
                
                vis_util.visualize_boxes_and_labels_on_image_array(
                    image00,
                    np.squeeze(boxes),
                    np.squeeze(classes).astype(np.int32),
                    np.squeeze(scores),
                    self.category_index,
                    use_normalized_coordinates=True,
                    line_thickness=3,
                    min_score_thresh=self.confident)
                #image00 = cv2.cvtColor(image00, cv2.COLOR_BGR2RGB)
                image = QtGui.QImage(image00, image00.shape[1], image00.shape[0], QtGui.QImage.Format_RGB888)
                self.maskdetectorlabel.setPixmap(QPixmap.fromImage(image))
                self.maskdetectorlabel.setScaledContents(True)
                
            

        
if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = mainwindow()
    window.show()
    sys.exit(app.exec_())