import numpy as np
import os
import tarfile
import tensorflow as tf
import cv2
from PIL import Image
from imageRotater import ImageRotater as IR

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util


class GymObjectDetector(object):
    videoPath = "/home/eamonn/FYP/Videos/" \
                "videoplayback"
    rotater = IR(videoPath)
    print("Rotation: " + IR.detectRotation())
    cap = cv2.VideoCapture(videoPath)
    gymObjects = {'Gym_Plate': {'Location': '',
                                'Frame': 0},
                  'FootWear': {'Location': [],
                                'Frame': 0}
                  }
    MODEL_PATH = '/home/eamonn/FYP/models/research/object_detection/'
    MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
    MODEL_FILE = MODEL_PATH + MODEL_NAME + '.tar.gz'
    PATH_TO_CKPT = MODEL_PATH + 'gym_plate_inference_graph' + '/frozen_inference_graph.pb'
    PATH_TO_LABELS = os.path.join(MODEL_PATH, 'training/object-detection.pbtxt')
    NUM_CLASSES = 4

    tar_file = tarfile.open(MODEL_FILE)
    for file in tar_file.getmembers():
        file_name = os.path.basename(file.name)
        if 'frozen_inference_graph.pb' in file_name:
            tar_file.extract(file, os.getcwd())

    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.\
        convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            plate_detected = False
            while True:
                ret, image_np = cap.read()
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
                    gymObjects['Gym_Plate']['Location'] = (np.squeeze(boxes)[0])
                    gymObjects['Gym_Plate']['Frame'] = (cap.get(cv2.CAP_PROP_POS_FRAMES))
                    cv2.destroyAllWindows()
                    break

    PATH_TO_CKPT = MODEL_PATH + 'FDG_inference_graph' + '/frozen_inference_graph.pb'

    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            plate_detected = False
            while True:
                ret, image_np = cap.read()
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
                if np.squeeze(scores)[0] > 0.5 and np.squeeze(classes).astype(np.int32)[0] == 4:
                    gymObjects['FootWear']['Location'] = (np.squeeze(boxes)[0])
                    gymObjects['FootWear']['Frame'] = (cap.get(cv2.CAP_PROP_POS_FRAMES))
                    break

    for object in ['Gym_Plate', 'FootWear']:
        normalisedCoordinates = gymObjects[object]['Location']
        if not rotater.toBeRotated:
            width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
            height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        else:
            width = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
            height = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        normalisedCoordinates[0] = normalisedCoordinates[0] * height
        normalisedCoordinates[1] = normalisedCoordinates[1] * width
        normalisedCoordinates[2] = normalisedCoordinates[2] * height
        normalisedCoordinates[3] = normalisedCoordinates[3] * width
        gymObjects[object]['Location'] = normalisedCoordinates





