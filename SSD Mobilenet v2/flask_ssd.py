import sys
import time
import argparse
from werkzeug.utils import secure_filename
import threading
import cv2
import pycuda.autoinit
import pycuda.driver as cuda
import threading
import os
import numpy as np

from utils.ssd_classes import get_cls_dict
from utils.ssd import TrtSSD
from utils.camera import add_camera_args, Camera
from utils.display import open_window, set_display, show_fps
from utils.visualization import BBoxVisualization

# Flask includes
from flask import Flask, render_template, send_from_directory, url_for, request, redirect

INPUT_HW = (300, 300)
# Create Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['ALLOWED_EXTENSIONS'] = set(['png','jpg','jpeg','mp4'])

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']

model = 'ssd_mobilenet_v2_cars'

class TrtThread(threading.Thread):
    
    def __init__(self,condition,vid,model,conf_th):
        threading.Thread.__init__(self)
        self.condition = condition
        self.vid = vid
        self.model = model
        self.conf_th = conf_th
        self.cuda_ctx = None
        self.trt_ssd = None
        self.running = False

    def run(self):
        global s_img, s_boxes, s_confs, s_clss
        print('TrtThread: loading the TRT SSD engine...')
        self.cuda_ctx = cuda.Device(0).make_context()  # GPU 0
        self.trt_ssd = TrtSSD(self.model, INPUT_HW)
        print('TrtThread: start running...')
        self.running = True
        while self.running:
            ret,img = self.vid.read()
            
            if img is not None:
                boxes, confs, clss = self.trt_ssd.detect(img, self.conf_th)
            with self.condition:
                s_img, s_boxes, s_confs, s_clss = img, boxes, confs, clss
                self.condition.notify()

        del self.trt_ssd
        self.cuda_ctx.pop()
        del self.cuda_ctx
        print('TrtThread: stopped...')

    def stop(self):
        self.running = False
        self.join()

def loop_and_display(condition, vis,wrvid):
    global s_img, s_boxes, s_confs, s_clss

    full_scrn = False
    fps = 0.0
    tic = time.time()
    while True:
        with condition:
            # Wait for the next frame and detection result.  When
            # getting the signal from the child thread, save the
            # references to the frame and detection result for
            # display.
            condition.wait()
            
            img, boxes, confs, clss = s_img, s_boxes, s_confs, s_clss
        if img is None:
            break
        
        img = vis.draw_bboxes(img, boxes, confs, clss)
        img = show_fps(img, fps)
        # cv2.imshow(WINDOW_NAME, img)
        wrvid.write(img)
        toc = time.time()
        curr_fps = 1.0 / (toc - tic)
        # calculate an exponentially decaying average of fps number
        fps = curr_fps if fps == 0.0 else (fps*0.95 + curr_fps*0.05)
        tic = toc
    return wrvid

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return redirect(url_for('uploaded_file',
                                filename=filename))

# Detect on video/image
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    FILEPATH = os.path.join(app.config['UPLOAD_FOLDER'],filename)
    cls_dict = get_cls_dict(model.split('_')[-1])
    vis = BBoxVisualization(cls_dict)
    cuda.init()

    if (filename.split('.')[-1] == 'mp4'):
        vid = cv2.VideoCapture(FILEPATH)
        if vid is None:
            print("NOT DETECTING VIDEO")
        output = os.path.join(app.config['UPLOAD_FOLDER'], filename.split('.')[0] + '_output.mp4')
        print(output)
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        out_format = 'mp4v'
        codec = cv2.VideoWriter_fourcc(*out_format)
        wrvid = cv2.VideoWriter(output, codec, fps, (width, height))

        condition = threading.Condition()
        trt_thread = TrtThread(condition,vid,model,conf_th=0.3)
        trt_thread.start()
        wrvid = loop_and_display(condition,vis,wrvid)
        trt_thread.stop()
        return send_from_directory(app.config['UPLOAD_FOLDER'], output.split('/')[-1])
    else :
        img = cv2.imread(FILEPATH)
        img = np.copy(img)
        cuda_ctx = cuda.Device(0).make_context()
        trt_ssd = TrtSSD(model, INPUT_HW)
        boxes,confs,clss = trt_ssd.detect(img,conf_th=0.3)
        img = vis.draw_bboxes(img,boxes,confs,clss)
        cuda_ctx.pop()
        del cuda_ctx
        cv2.imwrite('uploads/' + filename, img) 
        return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
