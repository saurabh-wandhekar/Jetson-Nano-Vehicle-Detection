# Instructions for evaluating accuracy (mAP) of SSD models

Preparation
-----------

1. Prepare image data and label ('bbox') file for the evaluation.  I used COCO [2017 Val images (5K/1GB)](http://images.cocodataset.org/zips/val2017.zip) and [2017 Train/Val annotations (241MB)](http://images.cocodataset.org/annotations/annotations_trainval2017.zip).  You could try to use your own dataset for evaluation, but you'd need to convert the labels into [COCO Object Detection ('bbox') format](http://cocodataset.org/#format-data) if you want to use code in this repository without modifications.

   More specifically, download the images and labels, and unzipped files into `${HOME}/data/coco/`.

   ```shell
   $ wget http://images.cocodataset.org/zips/val2017.zip \
          -O ${HOME}/Downloads/val2017.zip
   $ wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip \
          -O ${HOME}/Downloads/annotations_trainval2017.zip
   $ mkdir -p ${HOME}/data/coco/images
   $ cd ${HOME}/data/coco/images
   $ unzip ${HOME}/Downloads/val2017.zip
   $ cd ${HOME}/data/coco
   $ unzip ${HOME}/Downloads/annotations_trainval2017.zip
   ```

   Later on I would be using the following (unzipped) image and annotation files for the evaluation.

   ```
   ${HOME}/data/coco/images/val2017/*.jpg
   ${HOME}/data/coco/annotations/instances_val2017.json
   ```

2. Install 'pycocotools'.  The easiest way is to use `pip3 install`.

   ```shell
   $ sudo pip3 install pycocotools
   ```

   Alternatively, you could build and install it from [source](https://github.com/cocodataset/cocoapi).

3. Install additional requirements.

   ```shell
   $ sudo pip3 install progressbar2
   ```

Evaluation
----------

I've created the [eval_ssd.py](eval_ssd.py) script to do the [mAP evaluation](http://cocodataset.org/#detection-eval).

```
usage: eval_ssd.py [-h] [--mode {tf,trt}] [--imgs_dir IMGS_DIR]
                   [--annotations ANNOTATIONS]
                   {ssd_mobilenet_v2_cars}
```

The script takes 1 mandatory argument: 'ssd_mobilenet_v1_cars' .  In addition, it accepts the following options:

* `--mode {tf,trt}`: to evaluate either the unoptimized TensorFlow frozen inference graph (tf) or the optimized TensorRT engine (trt).
* `--imgs_dir IMGS_DIR`: to specify an alternative directory for reading image files.
* `--annotations ANNOTATIONS`: to specify an alternative annotation/label file.

```shell
$ python3 eval_ssd.py --mode trt ssd_mobilenet_v2_cars
```

