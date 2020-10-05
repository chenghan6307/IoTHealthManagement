import os
import cv2
import numpy as np
import tensorflow as tf
import sys
import time
import RPi.GPIO as GPIO

GPIO.setmode(GPIO.BOARD)
GPIO.setup(32, GPIO.OUT)
p = GPIO.PWM(32, 1)


#sys.path.append("..")

from utils import label_map_util
from utils import visualization_utils as vis_util

# Name of the directory containing the object detection module we're using
MODEL_NAME = 'inference_graph'

# Grab path to current working directory
CWD_PATH = os.getcwd()

# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,'labelmap.pbtxt')

IMAGE_NAME = 'temp.jpg'

# Path to image
PATH_TO_IMAGE = os.path.join(CWD_PATH,IMAGE_NAME)

# Number of classes the object detector can identify
NUM_CLASSES = 2

# Load the label map.
# Label maps map indices to category names, so that when our convolution
# network predicts `5`, we know that this corresponds to `king`.
# Here we use internal utility functions, but anything that returns a
# dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

confident = 0.999
# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)

  # Define input and output tensors (i.e. data) for the object detection classifier

  # Input tensor is the image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

  # Output tensors are the detection boxes, scores, and classes
  # Each box represents a part of the image where a particular object was detected
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

  # Each score represents level of confidence for each of the objects.
  # The score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

  # Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

# 選擇第二隻攝影機
cap = cv2.VideoCapture(0)

while(True):
  # 從攝影機擷取一張影像
  ret, frame = cap.read()

  # 顯示圖片
  cv2.imshow('frame', frame)
  
  if cv2.waitKey(1) & 0xFF == ord('p'):
  
    cv2.imwrite("temp.jpg", frame)
  
    

  # Load image using OpenCV and
  # expand image dimensions to have shape: [1, None, None, 3]
  # i.e. a single-column array, where each item in the column has the pixel RGB value
    image = cv2.imread(PATH_TO_IMAGE)
    image_expanded = np.expand_dims(image, axis=0)

  # Perform the actual detection by running the model with the image as input
    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: image_expanded})

  # Draw the results of the detection (aka 'visulaize the results')
    s_classes = classes[scores > confident]
    for i in range(len(s_classes)):
            if s_classes[i] in category_index.keys():
                class_name = category_index[s_classes[i]]['name']  # 得到英文class名稱
                if str(class_name) == 'good':
                  #print('warnning')
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
                  p.start(0.01)
                  p.ChangeFrequency(1318)
                  time.sleep(0.15)
                  p.ChangeFrequency(1175)
                  time.sleep(0.15)
                  p.ChangeFrequency(1046)
                  time.sleep(0.15)
                  p.stop()
                  
                  
                  
                  
    
    vis_util.visualize_boxes_and_labels_on_image_array(
        image,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=4,
        min_score_thresh=0.999)
    
  # All the results have been drawn on image. Now display the image.
    cv2.imshow('Object detector', image)

  # 若按下 q 鍵則離開迴圈
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

# 釋放攝影機
cap.release()


GPIO.cleanup()

# 關閉所有 OpenCV 視窗
cv2.destroyAllWindows()