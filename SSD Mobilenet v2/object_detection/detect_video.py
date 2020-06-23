"""
Sections of this code were taken from:
https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb
"""
# Import libraries
import numpy as np
import time
import os
import sys
import tensorflow as tf
import cv2
from absl import app, flags, logging
from absl.flags import FLAGS

# Import utilities
from utils import label_map_util
from utils import visualization_utils as vis_util


flags.DEFINE_integer('size', 300, 'resize images to')
flags.DEFINE_string('video', './input_video2.mp4',
                    'path to video file or number for webcam)')
flags.DEFINE_string('output', None, 'path to output video')
flags.DEFINE_string('output_format', 'mp4v', 'codec used in VideoWriter when saving video to file')

def main(_argv):

    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    for physical_device in physical_devices:
        tf.config.experimental.set_memory_growth(physical_device, True)

    # What model to use
    MODEL_NAME = 'ssd_mobilenet_v2_cars'

    # Path to frozen detection graph. This is the actual model that is used for the object detection.
    PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

    # List of the strings that is used to add correct label for each box.
    PATH_TO_LABELS = os.path.join('training', 'object-detection.pbtxt')

    NUM_CLASSES = 1

    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.compat.v1.GraphDef()
        with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    with detection_graph.as_default():
        with tf.compat.v1.Session(graph=detection_graph) as sess:
            # Definite input and output Tensors for detection_graph
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            # Each box represents a part of the image where a particular object was detected.
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    FPS=[]
            
    try:    
        vid = cv2.VideoCapture(int(FLAGS.video))
    except:  
        vid = cv2.VideoCapture(FLAGS.video)

    out = None

    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(vid.get(cv2.CAP_PROP_FPS))
    print('output video : %s'%FLAGS.output)
    if FLAGS.output:
        # by default VideoCapture returns float instead of int
        codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
        out = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))

    with detection_graph.as_default():
        with tf.compat.v1.Session(graph=detection_graph) as sess:
            while(vid.isOpened()):
                
                start_time=time.time()  # Start of inference time
                # Read the frame
                ret, frame = vid.read()
             
                if(ret==False):
                    break

                if frame is None:
                    logging.warning("Empty Frame")
                    time.sleep(0.1)
                    continue                
             
                color_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res_frame = cv2.resize(color_frame, (FLAGS.size,FLAGS.size))      # Resizing input frames
                image_np_expanded = np.expand_dims(res_frame, axis=0)
             
                # Actual detection.
                (boxes, scores, classes, num) = sess.run([detection_boxes, detection_scores, detection_classes, num_detections], feed_dict={image_tensor: image_np_expanded})

                # Visualization of the results of a detection.
                vis_util.visualize_boxes_and_labels_on_image_array(
                    res_frame, 
                    np.squeeze(boxes), 
                    np.squeeze(classes).astype(np.int32), 
                    np.squeeze(scores), 
                    category_index, 
                    use_normalized_coordinates=True, 
                    line_thickness=3,  
                    min_score_thresh=.30)
                
                end_time=time.time()   # End of inference time

                FPS.append(1/(end_time-start_time))
                FPS = FPS[-20:]

                color_frame = cv2.resize(res_frame, (width, height))
                
                color_frame = cv2.putText(color_frame, "FPS: {:.2f}".format(sum(FPS)/len(FPS)), (0, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
   
                output_rgb = cv2.cvtColor(color_frame, cv2.COLOR_RGB2BGR)
                cv2.imshow('output', output_rgb)
                if FLAGS.output:
                    out.write(output_rgb)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    out.release()
    vid.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
