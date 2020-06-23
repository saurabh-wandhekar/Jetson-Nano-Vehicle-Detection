# TensorRT

Example demonstrating how to optimize tensorflow model with TensorRT and run inferencing on NVIDIA Jetson or x86_64 PC platforms.  Highlights:  (The FPS numbers in this README are test results against JetPack 4.4, i.e. TensorRT 7, on Jetson Nano.)

* Run an optimized "ssd_mobilenet_v2" object detector ("trt_ssd_async.py") at 28~29 FPS on Jetson Nano.
* This should work on x86_64 PC with NVIDIA GPU(s) as well.  Some minor tweaks would be needed.  Please refer to [README_x86.md](https://github.com/jkjung-avt/tensorrt_demos/blob/master/README_x86.md) for more information.

Table of contents
-----------------

* [Prerequisite](#prerequisite)
* [Demo : SSD](#ssd)

<a name="prerequisite"></a>
Prerequisite
------------

The code in this repository was tested on Jetson Nano.  In order to run the demo below, first make sure you have the proper version of image (JetPack) installed on the target Jetson system.  For example, this is a blog post about setting up a Jetson Nano: [Setting up Jetson Nano: The Basics](https://jkjung-avt.github.io/setting-up-nano/).

More specifically, the target Jetson system must have TensorRT libraries installed.  **This demo would require TensorRT 5.x ~ 7.x.**

Furthermore, the demo program requires "cv2" (OpenCV) module for python3.  You could use the "cv2" module which came in the JetPack.

Lastly, if you plan to run this demo (SSD), you'd also need to have "tensorflowi-1.x" installed.  You could probably use the [official tensorflow wheels provided by NVIDIA](https://docs.nvidia.com/deeplearning/frameworks/pdf/Install-TensorFlow-Jetson-Platform.pdf).

In case you are setting up a Jetson Nano from scratch to run these demos, you could refer to the following blog posts.
* [JetPack-4.3 for Jetson Nano](https://jkjung-avt.github.io/jetpack-4.3/)
* [JetPack-4.4 for Jetson Nano](https://jkjung-avt.github.io/jetpack-4.4/)


<a name="ssd"></a>
Demo : SSD
------------

This demo shows how to convert pre-trained tensorflow Single-Shot Multibox Detector (SSD) models through UFF to TensorRT engines, and to do real-time object detection with the TensorRT engines.

NOTE: This particular demo requires TensorRT "Python API", which is only available in TensorRT 5.x+ on the Jetson systems.  In other words, this demo only works on Jetson systems properly set up with JetPack-4.2+, but **not** JetPack-3.x or earlier versions.

Assuming this repository has been cloned at "${HOME}/SSD_Mobilenet_v2/TensorRT", follow these steps:

1. Install requirements (pycuda, etc.) and build TensorRT engines from the pre-trained SSD models.

   ```shell
   $ cd ${HOME}/SSD_Mobilenet_v2/TensorRT/ssd
   $ ./install.sh
   $ ./build_engines.sh
   ```

2. Run the "trt_ssd.py" demo program.  The demo supports one model: "ssd_mobilenet_v2_cars". Note: It can be extended to other models.

**For image detection:**

   ```shell
   $ cd ${HOME}/SSD_Mobilenet_v2/TensorRT
   $ python3 trt_ssd.py --model ssd_mobilenet_v2_cars \
                        --image \
                        --filename test_images/1.jpg
                        --output_image output.jpg
   ```

**For video detection:**

   ```shell
   $ python3 trt_ssd.py --model ssd_mobilenet_v2_cars \
                        --file \
                        --filename test_videos/1.mp4
                        --output_file output.mp4
   ```

**Note**: The above code gives inference speed of around 20 FPS. Refer to point 3 for maximum inference speed.

3. I did implement an "async" version of ssd detection code which uses the concept of multithreading. When I tested "ssd_mobilenet_v2_cars" on the same car video(1.mp4) with the async demo program on the Jetson Nano DevKit, frame rate improved by 7-8 FPS.

   ```shell
   $ cd ${HOME}/SSD_Mobilenet_v2/TensorRT
   $ python3 trt_ssd_async.py --model ssd_mobilenet_v2_cars \
                              --file \
                              --filename test_videos/1.mp4
                              --output_file output.mp4
   ```

4. To verify accuracy (mAP) of the optimized TensorRT engines and make sure they do not degrade too much (due to reduced floating-point precision of "FP16") from the original TensorFlow frozen inference graphs, you could prepare validation data and run "eval_ssd.py".  Refer to [README_eval_ssd.md](README_eval_ssd.md) for details.

5. Check out these blog posts for implementation details:

   * [TensorRT UFF SSD](https://jkjung-avt.github.io/tensorrt-ssd/)
   * [Speeding Up TensorRT UFF SSD](https://jkjung-avt.github.io/speed-up-trt-ssd/)
   * [Verifying mAP of TensorRT Optimized SSD and YOLOv3 Models](https://jkjung-avt.github.io/trt-detection-map/)
   * Or if you'd like to learn how to train your own custom object detectors which could be easily converted to TensorRT engines and inferenced with "trt_ssd.py" and "trt_ssd_async.py": [Training a Hand Detector with TensorFlow Object Detection API](https://jkjung-avt.github.io/hand-detection-tutorial/)

6. To run inference on a local server, run the script flask_ssd.py, go to the address of the local server and upload an image or a video. The output image will be displayed on the server. We tried to upload the output video but it did not work. The outputs of image/video can be found locally in the uploads folder.

   ```shell
   $ python3 flask_ssd.py
   ```


Licenses
--------

The source code of [NVIDIA/TensorRT](https://github.com/NVIDIA/TensorRT) samples were referenced to develop the demo in this repository.  Those NVIDIA samples are under [Apache License 2.0](https://github.com/NVIDIA/TensorRT/blob/master/LICENSE).
