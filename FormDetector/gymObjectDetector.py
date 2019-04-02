import numpy as np
import os
import tarfile
import tensorflow as tf
import cv2
from imageRotater import ImageRotater as IR
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util


class GymObjectDetector(object):
    def __init__(self, objectsToBeTracked, videoPath):
        self.gymObjects = objectsToBeTracked
        self.videoPath = videoPath
        self.rotater = IR(self.videoPath)
        self.MODEL_PATH = '/home/eamonn/FYP/models/research/object_detection/'
        self.MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
        self.MODEL_FILE = self.MODEL_PATH + self.MODEL_NAME + '.tar.gz'
        self.PATH_TO_CKPT = self.MODEL_PATH + 'gym_plate_inference_graph' + '/frozen_inference_graph.pb'
        self.PATH_TO_LABELS = os.path.join(self.MODEL_PATH, 'training/object-detection.pbtxt')
        self.NUM_CLASSES = 4
        self.cap = cv2.VideoCapture(videoPath)

    def checkVideoOrientation(self):
        print("Rotation: " + self.rotater.detectRotation())

    def extractModel(self):
        tar_file = tarfile.open(self.MODEL_FILE)
        for file in tar_file.getmembers():
            file_name = os.path.basename(file.name)
            if 'frozen_inference_graph.pb' in file_name:
                tar_file.extract(file, os.getcwd())

    def importInferenceGraph(self):
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        return detection_graph

    def loadInLables(self):
        label_map = label_map_util.load_labelmap(self.PATH_TO_LABELS)
        categories = label_map_util. \
            convert_label_map_to_categories(label_map, max_num_classes=self.NUM_CLASSES, use_display_name=True)
        category_index = label_map_util.create_category_index(categories)
        return category_index

    def trackGymPlate(self):
        detection_graph = self.importInferenceGraph()
        category_index = self.loadInLables()
        with detection_graph.as_default():
            with tf.Session(graph=detection_graph) as sess:
                while True:
                    ret, image_np = self.cap.read()
                    image_np = self.rotater.rotateFrame(image_np)
                    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                    image_np_expanded = np.expand_dims(image_np, axis=0)
                    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                    # Each box represents a part of the image where a particular object was detected.
                    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                    # Each score represent how level of confidence for each of the objects.
                    # Score is shown on the result image, together with the class label.
                    scores = detection_graph.get_tensor_by_name('detection_scores:0')
                    classes = detection_graph.get_tensor_by_name('detection_classes:0')
                    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
                    # Actual detection.
                    (boxes, scores, classes, num_detections) = sess.run(
                        [boxes, scores, classes, num_detections],
                        feed_dict={image_tensor: image_np_expanded})
                    # Visualization of the results of a detection.
                    vis_util.visualize_boxes_and_labels_on_image_array(
                        image_np,
                        np.squeeze(boxes),
                        np.squeeze(classes).astype(np.int32),
                        np.squeeze(scores),
                        category_index,
                        use_normalized_coordinates=True,
                        line_thickness=8)
                    if np.squeeze(scores)[0] > 0.5:
                        self.gymObjects['Gym_Plate']['Location'] = (np.squeeze(boxes)[0])
                        self.gymObjects['Gym_Plate']['Frame'] = (self.cap.get(cv2.CAP_PROP_POS_FRAMES))
                        cv2.destroyAllWindows()
                        break

    def trackFootwear(self):
        found = False
        self.PATH_TO_CKPT = self.MODEL_PATH + 'FDG_inference_graph' + '/frozen_inference_graph.pb'
        detection_graph = self.importInferenceGraph()
        category_index = self.loadInLables()
        with detection_graph.as_default():
            with tf.Session(graph=detection_graph) as sess:
                while True:
                    ret, image_np = self.cap.read()
                    image_np = self.rotater.rotateFrame(image_np)
                    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                    image_np_expanded = np.expand_dims(image_np, axis=0)
                    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                    # Each box represents a part of the image where a particular object was detected.
                    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                    # Each score represent how level of confidence for each of the objects.
                    # Score is shown on the result image, together with the class label.
                    scores = detection_graph.get_tensor_by_name('detection_scores:0')
                    classes = detection_graph.get_tensor_by_name('detection_classes:0')
                    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
                    # Actual detection.
                    (boxes, scores, classes, num_detections) = sess.run(
                        [boxes, scores, classes, num_detections],
                        feed_dict={image_tensor: image_np_expanded})
                    # Visualization of the results of a detection.
                    vis_util.visualize_boxes_and_labels_on_image_array(
                        image_np,
                        np.squeeze(boxes),
                        np.squeeze(classes).astype(np.int32),
                        np.squeeze(scores),
                        category_index,
                        use_normalized_coordinates=True,
                        line_thickness=8)
                    for i, score in enumerate(np.squeeze(scores)):
                        if score<0.5:
                            break
                        if np.squeeze(classes).astype(np.int32)[i] == 4 and self.checkFootSize((np.squeeze(boxes)[i])):
                            found = True
                            self.gymObjects['FootWear']['Location'] = (np.squeeze(boxes)[0])
                            self.gymObjects['FootWear']['Frame'] = (self.cap.get(cv2.CAP_PROP_POS_FRAMES))
                            break
                    if found: break

    def trackPerson(self):
        self.PATH_TO_CKPT = self.MODEL_PATH + self.MODEL_NAME + '/frozen_inference_graph.pb'
        self.PATH_TO_LABELS = os.path.join(self.MODEL_PATH+'data', 'mscoco_label_map.pbtxt')
        detection_graph = self.importInferenceGraph()
        category_index = self.loadInLables()
        with detection_graph.as_default():
            with tf.Session(graph=detection_graph) as sess:
                while True:
                    ret, image_np = self.cap.read()
                    image_np = self.rotater.rotateFrame(image_np)
                    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                    image_np_expanded = np.expand_dims(image_np, axis=0)
                    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                    # Each box represents a part of the image where a particular object was detected.
                    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                    # Each score represent how level of confidence for each of the objects.
                    # Score is shown on the result image, together with the class label.
                    scores = detection_graph.get_tensor_by_name('detection_scores:0')
                    classes = detection_graph.get_tensor_by_name('detection_classes:0')
                    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
                    # Actual detection.
                    (boxes, scores, classes, num_detections) = sess.run(
                        [boxes, scores, classes, num_detections],
                        feed_dict={image_tensor: image_np_expanded})
                    # Visualization of the results of a detection.
                    vis_util.visualize_boxes_and_labels_on_image_array(
                        image_np,
                        np.squeeze(boxes),
                        np.squeeze(classes).astype(np.int32),
                        np.squeeze(scores),
                        category_index,
                        use_normalized_coordinates=True,
                        line_thickness=8)
                    if np.squeeze(scores)[0] > 0.5 and np.squeeze(classes).astype(np.int32)[0] == 1:
                        self.gymObjects['Person']['Location'] = (np.squeeze(boxes)[0])
                        self.gymObjects['Person']['Frame'] = (self.cap.get(cv2.CAP_PROP_POS_FRAMES))
                        break



    def checkFootSize(self, boxData):
        plateBoxData = self.gymObjects['Gym_Plate']['Location']
        footLength = (boxData[2]-boxData[0])
        footHeight = (boxData[3]-boxData[1])
        footarea = footHeight*footLength
        plateLength = (plateBoxData[2]-plateBoxData[0])
        plateHeight = (plateBoxData[3]-plateBoxData[1])
        plateArea = plateLength*plateHeight
        if footLength<plateLength and\
                footHeight<plateHeight and \
                footarea < plateArea*0.35:
            return True
        return False



    def normaliseLocations(self):
        for object in ['Gym_Plate', 'FootWear']:
            normalisedCoordinates = [1,2,3,4]
            coordinates = self.gymObjects[object]['Location']
            if not self.rotater.toBeRotated:
                width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
                height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            else:
                width = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
                height = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            normalisedCoordinates[0] = coordinates[1] * width  # x top left
            normalisedCoordinates[1] = coordinates[0] * height #y top left
            ymax = coordinates[2] * height
            xmax = coordinates[3] * width
            boxwidth = xmax - normalisedCoordinates[0]
            boxheight = ymax - normalisedCoordinates[1]
            normalisedCoordinates[2] = boxwidth
            normalisedCoordinates[3] = boxheight
            # x, y, w, h
            self.gymObjects[object]['Location'] = tuple(normalisedCoordinates)

    def getNormalisedObjectLocations(self):
        self.checkVideoOrientation()
        self.extractModel()
        self.trackGymPlate()
        self.trackFootwear()
        # self.trackPerson()
        self.normaliseLocations()
        return self.gymObjects
