# Vehicle-Detection-on-Jetson-Nano

Detection of cars by custom trained SSD Mobilenet v2 model implemented on Jetson Nano.

To install necessary libraries run:

```shell 
   $ pip install -r requirements.txt
```

### Training

The model has been trained with the official [TensorFlow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection). If you want to train your own model refer to [Tensorflow Object Detection API tutorial](https://pythonprogramming.net/introduction-use-tensorflow-object-detection-api-tutorial/). We custom trained the ssd_mobilenet_v2_coco model pretrained on the coco dataset.
You need **not** clone the [TensorFlow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection) repository. You can clone this repository itself and navigate to the SSD Mobilenet v2 directory and perform the steps mentioned in the tutorial.


### Inference

Navigate to the SSD Mobilenet v2 dir and follow the steps below:

1) Run the following commands in the terminal:

```shell
   $ protoc object_detection/protos/*.proto --python_out=.
   $ export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
```
**Note:** Skip to step 2 if there is no error in executing above commands.

If you get an error on the protoc command try running:
```shell
   $ protoc --version
```
If it is not the latest version download the latest version from the [protoc releases page](https://github.com/protocolbuffers/protobuf/releases). Download the python version, extract, navigate into the directory and then do:
```shell
   $ sudo ./configure
   $ sudo make check
   $ sudo make install
```
After that, try the protoc command again (again, make sure you are issuing this from the SSD Mobilenet v2 dir).

2) Run the following command(from within the SSD Mobilenet v2 directory) to set up the object_detection library:
```shell
   $ sudo python3 setup.py install
```

3) The model is saved in ssd_mobilenet_v2_cars dir inside object_detection dir. We will use this model for inference on images and videos/webcam.

   - **For inference on images run the detect.py script:**
     ```shell
        $ python3 detect.py --image test_images/1.png \
                            --output 1_output.png
     ```
  
   - **For inference on video/webcam run the detect_video.py script:(For webcam just replace the path of video with '0')**
     ```shell
        $ python3 detect_video.py --video test_videos/1.mp4 \  
                                  --output 1_output.mp4
     ```

4) The model gave an inference speed of 20 fps on CPU system(Intel i5 8th gen processor) and 30 fps on GPU system(Gefore GTX 1050 and Intel i5 8th gen processor).

5) This tensorflow model inferenced at only 4 fps on Jetson Nano. Tensorflow on Jetson Nano has memory leak issues. We converted our tensorflow model to TensorRT to improve performance. You can check the [TensorRT Readme](README_trt.md) for further instructions on how to convert to TensorRT and perform inference for TRT model.
   

