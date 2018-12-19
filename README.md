# Deep-6DPose: Recovering 6D Object Pose from a Single RGB Image

This is an implementation of [Deep-6D pose](https://arxiv.org/abs/1802.10367) on Python 3, Keras, and TensorFlow. The model generates bounding boxes, segmentation masks and 6 DoF pose for each instance of an object in the image. It's based on Feature Pyramid Network (FPN) and a ResNet101 backbone.

This is not a official implementation, and I forked and modified code from Mask R-CNN [https://github.com/matterport/Mask_RCNN]

I trained and tested with YCB Object Dataset [http://www.ycbbenchmarks.com/]
To generate the dataset, run the script below.

```
cd samples/ycb
python3 generateData.py
```

To train the model, run the script below.

```
cd samples/ycb
python3 ycb.py train --dataset=/path/to/ycb/dataset --weights=coco
```

To test the model, run the script below.

```
cd samples/ycb
python3 ycb.py mask --dataset=/path/to/ycb/dataset --weights=last
```

I did not implement the evaluation code.