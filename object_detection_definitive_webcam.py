import os
import cv2
import time
import argparse
import multiprocessing
import numpy as np
import tensorflow as tf
import threading

from utils import FPS, WebcamVideoStream
from multiprocessing import Process, Queue, Pool
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

from PIL import Image
import requests
from io import BytesIO
from io import StringIO

CWD_PATH = os.getcwd()
TF_MODELS_PATH = os.path.join(CWD_PATH, r'C:\Users\festevem\Documents\Control temperatura\TensorFlow Object Detection Models', 'trained_models')
# Path to frozen detection graph. This is the actual model that is used for the object detection
MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
"""Models available: 
    ssd_mobilenet_v1_coco_11_06_2017 -> Fast - 21 mAP
    ssd_inception_v2_coco_11_06_2017 -> Fast - 24 mAP
    rfcn_resnet101_coco_11_06_2017 -> Medium - 30 mAP
    faster_rcnn_resnet101_coco_11_06_2017 -> Medium - 32 mAP
    faster_rcnn_inception_resnet_v2_atrous_coco_11_06_2017 -> Slow - 37 mAP
"""
PATH_TO_CKPT = os.path.join(TF_MODELS_PATH, MODEL_NAME, 'frozen_inference_graph.pb')

# List of the strings that is used to add correct label for each box
PATH_TO_LABELS = os.path.join(CWD_PATH, r'C:\Users\festevem\Documents\Control temperatura\object_detection', 'data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90

class TotalPeople:
    i = 0

# Loading label map
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)

def model_load_into_memory():
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    return detection_graph

def detect_objects(image_np, sess, detection_graph):
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

    # Each box represents a part of the image where a particular object was detected
    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    scores = detection_graph.get_tensor_by_name('detection_scores:0')
    classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
    
    print(tf.global_variables())

    # Actual detection
    (boxes, scores, classes, num_detections) = sess.run(
        [boxes, scores, classes, num_detections],
        feed_dict={image_tensor: image_np_expanded})
    
    # Print every class + probability prediction
    classes_np = np.squeeze(classes).astype(np.int32)
    #print(classes_np)
    scores_np = np.squeeze(scores)
    #print("Scores 1\n")
    #print(scores_np)
    
    # Draw just class 1 detections (Person)
    total_people = 0 # Total observations per frame
    for i in range(classes_np.size):
        if classes_np[i]==1 and scores_np[i]>=0.5:
            total_people += 1
            TotalPeople.i = total_people
        elif classes_np[i] != 1:
            scores_np[i] = 0.02
    print("######################### " + str(TotalPeople.i) + " ########################")
    #print("Scores 2\n")
    #print(scores_np)
    
    # Visualization of the results of a detection
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        scores_np,
        category_index,
        use_normalized_coordinates=True,
        min_score_thresh=.5, 
        line_thickness=8)
    return image_np


    
    
    
    


from http.server import BaseHTTPRequestHandler, HTTPServer
import socketserver

class HandlerPacket(BaseHTTPRequestHandler):
    def _set_headers(self):
            self.send_response(200)
            self.send_header('Content-type', 'text/plain')
            self.end_headers()

    def do_GET(self):
            self._set_headers()
            message = str(TotalPeople.i)
            self.wfile.write(bytes(message, "utf8"))

class ServerHandlerPacket (threading.Thread):
   def __init__(self, name):
      threading.Thread.__init__(self)
      self.name = name
   def run(self):
       print('*****4 ' + str(TotalPeople.i))
       server_address = ('127.0.0.1', 8080)
       httpd = HTTPServer(server_address, HandlerPacket)
       print('Starting httpd...' + self.name)
       httpd.serve_forever()
       print('Ending httpd...' + self.name)

def main():
    video_capture = WebcamVideoStream(src=0,
                                      width=480,
                                      height=360).start()
    fps = FPS().start()
    
    detection_graph = model_load_into_memory()
    
    thread1 = ServerHandlerPacket("Thread-1-ServerHandlerPacket")
    thread1.daemon = True
    thread1.start()
    
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            while True:  
                # Camera detection loop
                frame = video_capture.read()
                cv2.imshow('Entrada', frame)
                t = time.time()
                output = detect_objects(frame, sess, detection_graph)
                cv2.imshow('Video', output)
                fps.update()
                print('[INFO] elapsed time: {:.2f}'.format(time.time() - t))
        
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
            video_capture.stop()
            fps.stop()
            print('[INFO] elapsed time (total): {:.2f}'.format(fps.elapsed()))
            print('[INFO] approx. FPS: {:.2f}'.format(fps.fps()))
        
            cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
    