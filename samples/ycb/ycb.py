"""
Mask R-CNN
Train on the toy ycb dataset.

Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 ycb.py train --dataset=/path/to/ycb/dataset --weights=coco

    # Resume training a model that you had trained earlier
    python3 ycb.py train --dataset=/path/to/ycb/dataset --weights=last

    # Train a new model starting from ImageNet weights
    python3 ycb.py train --dataset=/path/to/ycb/dataset --weights=imagenet

    # Test to a existed model 
    python3 ycb.py mask --dataset=/path/to/ycb/dataset --weights=last

"""

import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
#import imgaug
import cv2 as cv
from tqdm import tqdm
import scipy.io as sio

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils
from mrcnn import visualize
print(visualize.__file__)
# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

############################################################
#  Configurations
############################################################


class YcbConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "ycb"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 21 # Background + ycb

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 500

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9

    CLASS_NAMES = ['__background__', '002_master_chef_can', '003_cracker_box', '004_sugar_box',\
     '005_tomato_soup_can', '006_mustard_bottle', '007_tuna_fish_can', '008_pudding_box', \
     '009_gelatin_box', '010_potted_meat_can', '011_banana', '019_pitcher_base', '021_bleach_cleanser', \
     '024_bowl', '025_mug', '035_power_drill', '036_wood_block', '037_scissors', \
     '040_large_marker', '051_large_clamp', '052_extra_large_clamp', '061_foam_brick']


############################################################
#  Dataset
############################################################

class YcbDataset(utils.Dataset):

    def load_ycb(self, dataset_dir, subset):
        """Load a subset of the Ycb dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. Consider background is class number 0
        #for i, cl in enumerate(CLASS_NAMES[1:]):
        #    self.add_class('rgbd', i + 1, cl)
        self.add_class("ycb", 1, "002_master_chef_can")
        self.add_class("ycb", 2, "003_cracker_box")
        self.add_class("ycb", 3, "004_sugar_box")
        self.add_class("ycb", 4, "005_tomato_soup_can")
        self.add_class("ycb", 5, "006_mustard_bottle")
        self.add_class("ycb", 6, "007_tuna_fish_can")
        self.add_class("ycb", 7, "008_pudding_box")
        self.add_class("ycb", 8, "009_gelatin_box")
        self.add_class("ycb", 9, "010_potted_meat_can")
        self.add_class("ycb", 10, "011_banana")
        self.add_class("ycb", 11, "019_pitcher_base")
        self.add_class("ycb", 12, "021_bleach_cleanser") 
        self.add_class("ycb", 13, "024_bowl")
        self.add_class("ycb", 14, "025_mug")
        self.add_class("ycb", 15, "035_power_drill")
        self.add_class("ycb", 16, "036_wood_block")
        self.add_class("ycb", 17, "037_scissors")
        self.add_class("ycb", 18, "040_large_marker")
        self.add_class("ycb", 19, "051_large_clamp")
        self.add_class("ycb", 20, "052_extra_large_clamp")
        self.add_class("ycb", 21, "061_foam_brick")       
        # Train or validation dataset?
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)

        # Load annotations
        # VGG Image Annotator saves each image in the form:
        # { 'filename': '28503151_5b5b7ec140_b.jpg',
        #   'regions': {
        #       '0': {
        #           'region_attributes': {},
        #           'shape_attributes': {
        #               'all_points_x': [...],
        #               'all_points_y': [...],
        #               'name': 'polygon'}},
        #       ... more regions ...
        #   },
        #   'size': 100202
        # }
        # We mostly care about the x and y coordinates of each region
        annotations = json.load(open(os.path.join(dataset_dir, "via_region_data.json")))
        annotations = list(annotations.values())  # don't need the dict keys

        # The VIA tool saves images in the JSON even if they don't have any
        # annotations. Skip unannotated images.
        annotations = [a for a in annotations if a['regions']]

        # Add images
        for idx, a in tqdm(enumerate(annotations)):
            # Get the x, y coordinaets of points of the polygons that make up
            # the outline of each object instance. There are stores in the
            # shape_attributes (see json format above)
            polygons = [r['shape_attributes'] for r in a['regions'].values()]
            objects = [s['region_attributes'] for s in a['regions'].values()]
            num_ids = [int(n['ycb']) for n in objects]
            poses = [t['pose_attributes'] for t in a['regions'].values()]
            #num_ids = 0
            # load_mask() needs the image size to convert polygons to masks.
            # Unfortunately, VIA doesn't include it in JSON, so we must read
            # the image. This is only managable since the dataset is tiny.
            #image_path = os.path.join(dataset_dir, a['filename'])
            image_path = os.path.join(dataset_dir, a['filename'] + '.jpg')
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]
            self.add_image(
                "ycb",
                image_id=a['filename'],  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                polygons=polygons,
                poses=poses,
                num_ids=num_ids)

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a balloon dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "ycb":
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        
        info = self.image_info[image_id]
        num_ids = info['num_ids']
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1
        num_ids = np.array(num_ids, dtype=np.int32)
        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        return mask.astype(np.bool), num_ids

    def load_pose(self, image_id):
        """Generate poses for an image.
       Returns:
        poses: A bool array of shape [4, instance count] with pose information [rx, ry, rz, tz] per instance.
        class_ids: a 1D array of class IDs of the pose.
        """
        # If not a ycb dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "ycb":
            return super(self.__class__, self).load_pose(image_id)

        # Convert polygons to a bitmap mask of shape
        # [4, instance_count]
        
        info = self.image_info[image_id]
        num_ids = info['num_ids']
        pose = np.zeros([len(info['poses']), 4], dtype=np.float32)
        #pose = np.zeros([4, len(info['poses'])], dtype=np.float32)
        for i, p in enumerate(info['poses']):
            pose[i, 0] = p['rx']
            pose[i, 1] = p['ry']
            pose[i, 2] = p['rz']
            pose[i, 3] = p['tz']

        num_ids = np.array(num_ids, dtype=np.int32)
        return pose, num_ids
        

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "ycb":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = YcbDataset()
    dataset_train.load_ycb(args.dataset, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = YcbDataset()
    dataset_val.load_ycb(args.dataset, "val")
    dataset_val.prepare()

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    #augmentation = imgaug.augmenters.Sometimes(0.5, [
    #    imgaug.augmenters.Fliplr(0.5),
    #    imgaug.augmenters.GaussianBlur(sigma=(0.0, 5.0))
    #])    
    # Training - Stage 1
    print("Training network heads")
    
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=200,
                layers='heads',
                augmentation=None)
    # Training - Stage 2
    # Finetune layers from ResNet stage 4 and up
    print("Fine tune Resnet stage 4 and up")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=400,
                layers='3+',
                augmentation=None)
    
    # Training - Stage 3
    # Fine tune all layers
    print("Fine tune all layers")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE / 10.0,
                epochs=500,
                layers='5+',
                augmentation=None)
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE / 100.0,
                epochs=600,
                layers='all', 
                augmentation=None)


def color_splash(image, mask):
    """Apply color splash effect.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]

    Returns result image.
    """
    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    # We're treating all instances as one, so collapse the mask into one layer
    mask = (np.sum(mask, -1, keepdims=True) >= 1)
    # Copy color pixels from the original color image where mask is set
    if mask.shape[0] > 0:
        splash = np.where(mask, image, gray).astype(np.uint8)
    else:
        splash = gray
    return splash


def detect_and_color_splash(model, image_path=None, video_path=None):
    assert image_path or video_path

    # Image or video?
    if image_path:
        # Run model detection and generate the color splash effect
        print("Running on {}".format(args.image))
        # Read image
        image = skimage.io.imread(args.image)
        # Detect objects
        r = model.detect([image], verbose=1)[0]
        # Color splash
        splash = color_splash(image, r['masks'])
        # Save output
        file_name = "splash_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
        skimage.io.imsave(file_name, splash)
    elif video_path:
        import cv2
        # Video capture
        vcapture = cv2.VideoCapture(video_path)
        width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = vcapture.get(cv2.CAP_PROP_FPS)

        # Define codec and create video writer
        file_name = "splash_{:%Y%m%dT%H%M%S}.avi".format(datetime.datetime.now())
        vwriter = cv2.VideoWriter(file_name,
                                  cv2.VideoWriter_fourcc(*'MJPG'),
                                  fps, (width, height))

        count = 0
        success = True
        while success:
            print("frame: ", count)
            # Read next image
            success, image = vcapture.read()
            if success:
                # OpenCV returns images as BGR, convert to RGB
                image = image[..., ::-1]
                # Detect objects
                r = model.detect([image], verbose=0)[0]
                # Color splash
                splash = color_splash(image, r['masks'])
                # RGB -> BGR to save image to video
                splash = splash[..., ::-1]
                # Add image to video writer
                vwriter.write(splash)
                count += 1
        vwriter.release()
    print("Saved to ", file_name)

def project(img, object3d, Rt, intrinsics, color):
    min_x = min(object3d[:, 0])
    min_y = min(object3d[:, 1])
    min_z = min(object3d[:, 2])
    max_x = max(object3d[:, 0])
    max_y = max(object3d[:, 1])
    max_z = max(object3d[:, 2])
    bb3d = np.zeros((8, 4), dtype=float)
    ##0-1, 0-2, 0-3, 7-6, 7-5, 7-4, 1-3, 1-6, 2-4, 2-6, 3-4, 3-5
    bb3d[0, :] = [min_x, min_y, min_z, 1.]
    bb3d[1, :] = [min_x, min_y, max_z, 1.]
    bb3d[2, :] = [min_x, max_y, min_z, 1.]
    bb3d[3, :] = [max_x, min_y, min_z, 1.]
    bb3d[4, :] = [max_x, max_y, min_z, 1.]
    bb3d[5, :] = [max_x, min_y, max_z, 1.]
    bb3d[6, :] = [min_x, max_y, max_z, 1.]
    bb3d[7, :] = [max_x, max_y, max_z, 1.]
    cam2d = np.dot(intrinsics, np.dot(Rt, bb3d.transpose()))
    cam2d[0, :] = cam2d[0, :] / cam2d[2, :]
    cam2d[1, :] = cam2d[1, :] / cam2d[2, :]
    cam2d = cam2d.astype(np.int32)
    #cam2d = np.dot(intrinsics, np.dot(Rt, np.hstack((object3d, np.ones((object3d.shape[0], 1), dtype=float))).transpose()))
    cv.line(img, (cam2d[0, 0], cam2d[1, 0]), (cam2d[0, 1], cam2d[1, 1]), (255 * color[0], 255 * color[1], 255 * color[2]))
    cv.line(img, (cam2d[0, 0], cam2d[1, 0]), (cam2d[0, 2], cam2d[1, 2]), (255 * color[0], 255 * color[1], 255 * color[2]))
    cv.line(img, (cam2d[0, 0], cam2d[1, 0]), (cam2d[0, 3], cam2d[1, 3]), (255 * color[0], 255 * color[1], 255 * color[2]))
    cv.line(img, (cam2d[0, 7], cam2d[1, 7]), (cam2d[0, 6], cam2d[1, 6]), (255 * color[0], 255 * color[1], 255 * color[2]))
    cv.line(img, (cam2d[0, 7], cam2d[1, 7]), (cam2d[0, 5], cam2d[1, 5]), (255 * color[0], 255 * color[1], 255 * color[2]))
    cv.line(img, (cam2d[0, 7], cam2d[1, 7]), (cam2d[0, 4], cam2d[1, 4]), (255 * color[0], 255 * color[1], 255 * color[2]))
    cv.line(img, (cam2d[0, 1], cam2d[1, 1]), (cam2d[0, 5], cam2d[1, 5]), (255 * color[0], 255 * color[1], 255 * color[2]))
    cv.line(img, (cam2d[0, 1], cam2d[1, 1]), (cam2d[0, 6], cam2d[1, 6]), (255 * color[0], 255 * color[1], 255 * color[2]))
    cv.line(img, (cam2d[0, 2], cam2d[1, 2]), (cam2d[0, 4], cam2d[1, 4]), (255 * color[0], 255 * color[1], 255 * color[2]))
    cv.line(img, (cam2d[0, 2], cam2d[1, 2]), (cam2d[0, 6], cam2d[1, 6]), (255 * color[0], 255 * color[1], 255 * color[2]))
    cv.line(img, (cam2d[0, 3], cam2d[1, 3]), (cam2d[0, 4], cam2d[1, 4]), (255 * color[0], 255 * color[1], 255 * color[2]))
    cv.line(img, (cam2d[0, 3], cam2d[1, 3]), (cam2d[0, 5], cam2d[1, 5]), (255 * color[0], 255 * color[1], 255 * color[2]))

    #cam2d_obj = np.dot(intrinsics, np.dot(Rt, np.hstack((object3d, np.ones((object3d.shape[0], 1), dtype=float))).transpose()))
    #cam2d_obj[0, :] = cam2d_obj[0, :] / cam2d_obj[2, :]
    #cam2d_obj[1, :] = cam2d_obj[1, :] / cam2d_obj[2, :]
    #cam2d_obj = cam2d_obj.astype(np.int32)
    #centroid_color = np.array((int(color[0] * 255.0),int(color[1] * 255.0),int(color[2] * 255.0)))
    #for i in range(cam2d_obj.shape[1]):
    #    if cam2d_obj[1, i] >= 0 and cam2d_obj[1, i] <= cam2d_obj.shape[1] and cam2d_obj[0, i] >= 0 and cam2d_obj[1, i] <= cam2d_obj.shape[0]:
    #        img[cam2d_obj[1, i] - 1, cam2d_obj[0, i] - 1, 0] = 255
    #        img[cam2d_obj[1, i] - 1, cam2d_obj[0, i] - 1, 1] = 255
    #        img[cam2d_obj[1, i] - 1, cam2d_obj[0, i] - 1, 2] = 255
    #        #cv.circle(img, (cam2d_obj[0, i], cam2d_obj[1, i]), 2, color)    


def visualize3DBB(img, img_file, pred_pose, pred_bbox, pred_classes, objects3D, colors):
    
    dataset_dir = './dataset/val/'
    meta_path = os.path.join(dataset_dir, img_file[:-4] + '.mat')
    meta_info = sio.loadmat(meta_path)
    intrinsics = meta_info['intrinsic_matrix']
    poses = meta_info['poses']
    # Number of instances
    N = pred_bbox.shape[0]
    #colors = random_colors(N)
    if not N:
        print("\n*** No instances to display *** \n")
        return 


    for i in range(N):
        Rt = np.zeros((3, 4), np.float32)
        #Rt = np.eye(4, np.float32)
        class_id = pred_classes[i]
        y1, x1, y2, x2 = pred_bbox[i]
        rx, ry, rz, tz = pred_pose[i, class_id, :]
        rotation_mat, _ = cv.Rodrigues(pred_pose[i, class_id, :-1])
        Rt[:, :3] = rotation_mat
        u0 = (x1 + x2) / 2.
        v0 = (y1 + y2) / 2.
        cx = intrinsics[0, -1]
        cy = intrinsics[1, -1]
        fx = intrinsics[0, 0]
        fy = intrinsics[1, 1]
        tx = (u0 - cx) * tz / fx
        ty = (v0 - cy) * tz / fy
        Rt[0, -1] = tx
        Rt[1, -1] = ty
        Rt[2, -1] = tz
        color = colors[class_id]
        project(img, objects3D[class_id - 1], Rt, intrinsics, color)







def detect_and_mask(config, model, image_path=None, image_dir=None, video_path=None, output_dir=None):
    assert image_path or video_path or image_dir
    colors = visualize.random_colors(config.NUM_CLASSES)
    object_list = np.genfromtxt('./dataset/obj_list.txt', dtype=str, delimiter=',')
    obj_dir = object_dir = './dataset/models'
    objects_pointcloud = []
    for obj in object_list:
        obj_path = os.path.join(object_dir, obj, 'points.xyz')
        objects = np.loadtxt(obj_path, delimiter=' ', dtype=float)
        objects_pointcloud.append(objects)

    # Image or video?
    if image_path:
        # Run model detection and generate the color splash effect
        print("Running on {}".format(args.image))
        # Read image
        image = skimage.io.imread(args.image)
        # Detect objects
        r = model.detect([image], verbose=1)[0]
        # Color splash
        masked_image = visualize.mask_instances(image, r['rois'], r['masks'], r['class_ids'], config.CLASS_NAMES, r['scores'])
        # Save output
        file_name = os.path.join(output_dir, "mask_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now()))
        skimage.io.imsave(file_name, masked_image)
    elif image_dir:
        image_list = os.listdir(image_dir)
        print(image_list[0])
        image_list = sorted(image_list)
        print(config.CLASS_NAMES)
        for file in image_list:
            if 'jpg' in file:
                print(file)
                image = cv.imread(os.path.join(image_dir, file))
                image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
                r = model.detect([image], verbose=1)[0]
                #visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], config.CLASS_NAMES, r['scores'])
                masked_image = visualize.mask_instances(image, r['rois'], r['masks'], r['class_ids'], config.CLASS_NAMES, colors, r['scores'])
                pose_image = image.copy()
                visualize3DBB(pose_image, file, r['poses'], r['rois'], r['class_ids'], objects_pointcloud, colors)
                cv.imshow("", cv.cvtColor(masked_image, cv.COLOR_RGB2BGR))
                cv.imshow("pose", cv.cvtColor(pose_image, cv.COLOR_RGB2BGR))
                cv.waitKey(1)
                file_name = os.path.join(output_dir, file)
                cv.imwrite(file_name, cv.hconcat([cv.cvtColor(masked_image, cv.COLOR_RGB2BGR), cv.cvtColor(pose_image, cv.COLOR_RGB2BGR)]))
                #skimage.io.imsave(file_name, masked_image)
    elif video_path:
        import cv2
        # Video capture
        vcapture = cv2.VideoCapture(video_path)
        width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = vcapture.get(cv2.CAP_PROP_FPS)

        # Define codec and create video writer
        file_name = "splash_{:%Y%m%dT%H%M%S}.avi".format(datetime.datetime.now())
        vwriter = cv2.VideoWriter(file_name,
                                  cv2.VideoWriter_fourcc(*'MJPG'),
                                  fps, (width, height))

        count = 0
        success = True
        while success:
            print("frame: ", count)
            # Read next image
            success, image = vcapture.read()
            if success:
                # OpenCV returns images as BGR, convert to RGB
                image = image[..., ::-1]
                # Detect objects
                r = model.detect([image], verbose=0)[0]
                # Color splash
                splash = color_splash(image, r['masks'])
                # RGB -> BGR to save image to video
                splash = splash[..., ::-1]
                # Add image to video writer
                vwriter.write(splash)
                count += 1
        vwriter.release()
    print("Saved to ", file_name)


############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect ycbs.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'splash' or 'mask'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/ycb/dataset/",
                        help='Directory of the ycb dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to apply the color splash effect on')
    parser.add_argument('--video', required=False,
                        metavar="path or URL to video",
                        help='Video to apply the color splash effect on')
    parser.add_argument('--img_dir', required=False,
                        metavar="/path/to/imagedir/",
                        help='path to image directory')
    parser.add_argument('--output', required=False,
                        metavar="/path/to/output/",
                        help='path to output directory')
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "splash":
        assert args.image or args.video,\
               "Provide --image or --video to apply color splash"
    elif args.command == 'mask':
        assert args.image or args.video or args.img_dir, "Provide --image or --video or --img_dir to apply mask"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = YcbConfig()
    else:
        class InferenceConfig(YcbConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()[1]
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask", "mrcnn_pose"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model)
    elif args.command == "splash":
        detect_and_color_splash(model, image_path=args.image,video_path=args.video)
    elif args.command == 'mask':
        detect_and_mask(config, model, image_dir=args.img_dir, output_dir=args.output)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'splash'".format(args.command))
