"""
Mask R-CNN
Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
Edited by Christopher Conrady

--- version history ---

[1] v1.001
- 'detect' command creates bbox, mask and centroid, using open-cv (replaces 'splash').

[2] v1.002
- object tracking on video

[3] v1.003
- object tracking (with N frame(s) memory) to reduce false-positive tracking

[4] v1.004
- improve cv2 visual output (alpha mask, solid box, centroid & text)

v1.004.1
- fix mask bug (missing masks)

v1.004.2
- fix object tracking bug (pop logic error)

v1.004.3
- command line argument (option) to disable object tracking

[5] v1.005.0
- Added training instructions
- *Fixed bug on command train, ~BalloonDataset~ to FishDataset
- Added model parameters from config.py
- Added training schedule, heads for 20, then convs for 40
- Added augmentation on train & val
*NB

[6] v1.006.0 (---IGNORE---)
- Some changes for mAP calcs. (---IGNORE---)

[7] v1.007.0 
- FishInferenceConfig at top is now set to be used.
  (Before, the class InferenceConfig was created for inference at bottom.)
- Added "test" to allowed datasets for inference
- Changed augmentation sequence

[8] v1.008.0 
- Object tracking on video now produces a .csv file of tracked objects.

"""

# --- Usage ---

# --- TRAIN ---
# [i.] Train a new model starting from pre-trained COCO weights
# python fish.py train --dataset=\path\to\fish\dataset --weights=coco
# e.g. python fish.py train --dataset=C:\Users\chris\Desktop\fish1 --weights=coco --logs E:\logs
# e.g. python fish.py train --dataset=..\..\datasets\fish\fish2A --weights=coco --logs E:\logs

# [ii.] Resume training a model from previous weights
# python fish.py train --dataset=\path\to\fish\dataset --weights=last
# e.g. python fish.py train --dataset=C:\Users\chris\Desktop\fish1 --weights=E:\logs\fish20201231T1053\mask_rcnn_fish_0003.h5 --logs E:\logs\fish20201231T1053

# --- DETECT ---
# [i.]  Detection (bbox, mask, centroid) - *IMAGE* (folder)
# python fish.py detect --weights=C:\Users\Chris\Desktop\offline\maskrcnn-3\Mask_RCNN\logs\fish\mask_rcnn_fish_0025.h5 --image=C:\Users\Chris\Desktop\offline\maskrcnn-3\Mask_RCNN\datasets\fish\sample\

# [ii.] Detection (bbox, mask, centroid) - *VIDEO*
# python fish.py detect --weights=C:\Users\Chris\Desktop\offline\maskrcnn-3\Mask_RCNN\logs\fish\mask_rcnn_fish_0025.h5 --video=C:\Users\Chris\Desktop\offline\maskrcnn-3\Mask_RCNN\datasets\fish\video\test_min_5s.MP4

# [ii.] Detection (bbox, mask, centroid) - *VIDEO* (with centroid tracking)
# python fish.py detect --weights=C:\Users\Chris\Desktop\offline\maskrcnn-3\Mask_RCNN\_weights\mask_rcnn_fish_0047.h5 --video=C:\Users\Chris\Dropbox\__SYNC__\1_results\test_min_28s.MP4 --tracking Y



_VERSION = "v1.008.0"
if __name__ == '__main__':
    print("\n--- ", _VERSION, " ---\n")

import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
import cv2 # v1.001
from pyimagesearch.centroidtracker import CentroidTracker # v1.002
from skimage.measure import find_contours # v1.004
from imgaug import augmenters as iaa # v1.005.0

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

############################################################
#  Configurations
############################################################


class FishConfig(Config):
    """Configuration for training on the toy dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "fish"

    # >>> ----- Memory vs. Accuracy vs. Training Time ----- <<<
    # >>> ------------------ (trade-offs) ----------------- <<<
    # https://github.com/matterport/Mask_RCNN/wiki

    # [1]
    # Backbone network architecture
    # Supported values are: resnet50, resnet101.
    # You can also provide a callable that should have the signature
    # of model.resnet_graph. If you do so, you need to supply a callable
    # to COMPUTE_BACKBONE_SHAPE as well
    BACKBONE = "resnet50"

    # [2]
    # layers: Allows selecting wich layers to train. It can be:
    #   - A regular expression to match layer names to train
    #   - One of these predefined values (in order or memory usage, desc):
    #     all: All the layers
    #     3+: Train Resnet stage 3 and up
    #     4+: Train Resnet stage 4 and up
    #     5+: Train Resnet stage 5 and up
    #     heads: The RPN, classifier and mask heads of the network
      
        # # From a specific Resnet stage and up
        # "all": ".*",
        # "3+": r"(res3.*)|(bn3.*)|(res4.*)|(bn4.*)|(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
        # "4+": r"(res4.*)|(bn4.*)|(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
        # "5+": r"(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
        # "heads": r"(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",        
    # 
    # SEARCH: QLAYERS

    # [3]
    # Setting image resolutions to a smaller size reduces memory
    # requirements and cuts training and inference times as well.
    # IMAGE_MIN_DIM = 800
    # IMAGE_MAX_DIM = 1024

    IMAGE_MIN_DIM = 460
    IMAGE_MAX_DIM = 576

    # IMAGE_MIN_DIM = 1080
    # IMAGE_MAX_DIM = 1920
    # IMAGE_RESIZE_MODE = "square" # "none"

    # [4]
    # Number of images to train with on each GPU. A 12GB GPU can typically
    # handle 2 images of 1024x1024px.
    # Adjust based on your GPU memory and image sizes. Use the highest
    # number that your GPU can handle for best performance.
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # [5]
    # Number of ROIs per image to feed to classifier/mask heads
    # The Mask RCNN paper uses 512 but often the RPN doesn't generate
    # enough positive proposals to fill this and keep a positive:negative
    # ratio of 1:3. You can increase the number of proposals by adjusting
    # the RPN NMS threshold.
    # Use fewer ROIs in training the second stage.
    # This setting is like the batch size for the seconds stage
    # of the model.
    TRAIN_ROIS_PER_IMAGE = 100 # 200

    # [6] Maximum number of ground truth instances to use in one image
    # Reduce the maximum number of instances per image if your images
    # don't have a lot of objects.
    MAX_GT_INSTANCES = 10 # 100


    # >>> ------------------ (other) ----------------- <<<
    # Number of training steps per epoch
    STEPS_PER_EPOCH = 300

    # Number of validation steps to run at the end of every training epoch.
    # A bigger number improves accuracy of validation stats, but slows
    # down the training.
    VALIDATION_STEPS = 100


    # >>> ------------------ (model params) ----------------- <<<
    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + fish

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9

# Actioned: HERE_A #v1.007.0 
class FishInferenceConfig(FishConfig):
    # Set batch size to 1 to run one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    # Don't resize imager for inferencing
    # IMAGE_RESIZE_MODE = "pad64" (causes an error, both sides must be div by 64)
    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    # RPN_NMS_THRESHOLD = 0.7

############################################################
#  Dataset
############################################################

class FishDataset(utils.Dataset):

    def load_fish(self, dataset_dir, subset):
        """Load a subset of the Fish dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have only one class to add.
        self.add_class("fish", 1, "chrysoblephus_laticeps")

        # Train or validation dataset?
        assert subset in ["train", "val", "test"] # v1.007.0
        dataset_dir = os.path.join(dataset_dir, subset)

        # Load annotations
        # VGG Image Annotator (up to version 1.6) saves each image in the form:
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
        # Note: In VIA 2.0, regions was changed from a dict to a list.

        annotations = json.load(open(os.path.join(dataset_dir, "0_via_region_data.json"))) # <--- v1.0.005
        # annotations = json.load(open(os.path.join(dataset_dir, "via_region_data.json"))) 
        annotations = list(annotations.values())  # don't need the dict keys

        # The VIA tool saves images in the JSON even if they don't have any
        # annotations. Skip unannotated images.
        annotations = [a for a in annotations if a['regions']]

        # Add images
        for a in annotations:
            # Get the x, y coordinaets of points of the polygons that make up
            # the outline of each object instance. These are stores in the
            # shape_attributes (see json format above)
            # The if condition is needed to support VIA versions 1.x and 2.x.
            if type(a['regions']) is dict:
                polygons = [r['shape_attributes'] for r in a['regions'].values()]
            else:
                polygons = [r['shape_attributes'] for r in a['regions']] 

            # load_mask() needs the image size to convert polygons to masks.
            # Unfortunately, VIA doesn't include it in JSON, so we must read
            # the image. This is only managable since the dataset is tiny.
            image_path = os.path.join(dataset_dir, a['filename'])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            self.add_image(
                "fish",
                image_id=a['filename'],  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                polygons=polygons)

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a fish dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "fish":
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "fish":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = FishDataset()
    dataset_train.load_fish(args.dataset, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = FishDataset() # <--- v1.0.005
    dataset_val.load_fish(args.dataset, "val")
    dataset_val.prepare()

    # Image augmentation
    # https://imgaug.readthedocs.io/en/latest/source/overview_of_augmenters.html
    # v1.007.0  #Sometimes (I think this applies to every image.)
    augmentation = iaa.SomeOf((0, 2), [
	    # [1] Flip 50%(x) of all images horizontally
	    iaa.Fliplr() 
	    # [2] Flip %50(x) of all images vertically
	    ,iaa.Flipud() 
	    # [3] Rotate images by (x1) to (x2) degrees:
	    ,iaa.Affine(rotate=(-45, 45))
	    # [4] Multiply all pixels in an image with a specific value,
	    # ... in other words, make the image darker or brighter.
	    ,iaa.Multiply((0.5, 1.5)) 
	    # [5] Apply blur
	    ,iaa.AverageBlur(k=((25, 50), (10, 30)))
	    # [6] Add noise (Gaussian)
	    ,iaa.AdditiveGaussianNoise(scale=(0, 0.1*255))
	], random_order=True)

    # v1.005 --->
    # *** This training schedule is an example. Update to your needs ***
    # SEARCH: QLAYERS
    # If starting from imagenet, train heads only for a bit
    # since they have random weights

    # print("...training model...(heads)")
    # model.train(dataset_train, dataset_val,
    #             learning_rate=config.LEARNING_RATE,
    #             epochs=2,
    #             augmentation=augmentation,
    #             layers='heads')

    print("...training model...(all)")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=1,
                augmentation=augmentation,
                layers='heads')
    # <--- v1.0.005


# v1.004 --->

# v1.004.1 --->
# def format_mask(mask):
#     padded_mask = np.zeros(
#         (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
#     padded_mask[1:-1, 1:-1] = mask
#     contours = find_contours(padded_mask, 0.5)
#     verts = contours[-1]
#     # Subtract the padding and flip (y, x) to (x, y)
#     verts = np.fliplr(verts) - 1
#     # Additional formatting for open-cv
#     verts = verts.astype(np.int32)
#     verts = verts.reshape((-1,1,2))
#     return verts
# v1.004.1 <---

# v1.004.1 --->
def apply_cv_masks(image, overlay, mask, color=(0, 0, 255)):
    img, overlay = image.copy(), overlay.copy()
    padded_mask = np.zeros(
        (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
    padded_mask[1:-1, 1:-1] = mask
    contours = find_contours(padded_mask, 0.5)
    for verts in contours:
        # Subtract the padding and flip (y, x) to (x, y)
        verts = np.fliplr(verts) - 1
        # Additional formatting for open-cv
        verts = verts.astype(np.int32)
        verts = verts.reshape((-1,1,2))
        cv2.polylines(img, [verts], True, color)
        cv2.fillPoly(overlay, pts=[verts], color=color)
    return (img, overlay)
# v1.004.1 <---
    
def apply_cv_boxes(image, box, color=(0, 0, 255)):
    img = image.copy()
    startY, startX, endY, endX = box
    cv2.rectangle(img, (startX, startY), (endX, endY), color, 1)
    return img

def apply_cv_centroid(image, centroid, color=(0, 0, 255)):
    img = image.copy()
    (cX, cY) = centroid
    cv2.circle(img, (cX, cY), 4, color, -1)
    return img
            
def apply_cv_text(image, text, x, y, color=(0, 0, 255)):
    img = image.copy()
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return img

def np_pop(np_lst, i):
    lst = list(np_lst)
    lst.pop(i)
    return np.array(lst, dtype=np_lst.dtype)

def apply_frame_notations(image, masks=[], boxes=[], centroids=[], class_ids=[], scores=[]):
    N = len(class_ids)
    output = image.copy()
    alpha_layer = image.copy()
    if N > 0:
        for i in range(N):
            # [1] apply mask
            if len(masks):
                (output, alpha_layer) = apply_cv_masks(output, alpha_layer, masks[:, :, i])

            # [2] apply bounding box
            if len(boxes):
                output = apply_cv_boxes(output, boxes[i])

            # [3] apply centroid
            if len(centroids):
                output = apply_cv_centroid(output, centroids[i])

            # [4] apply text annotations (class & class probability)
            if all([len(class_ids), len(scores)]):
                annotation = "{}: {}".format(class_ids[i], scores[i])
                y, x, b, a = boxes[i]
                output = apply_cv_text(output, annotation, x, y-5)
        
        # [1] apply mask (overlay)
        if len(masks):
            alpha = 0.3
            cv2.addWeighted(alpha_layer, alpha, output, 1 - alpha, 0, output)
    return output
# <--- v1.004

# v1.001 + v1.002 + v1.004 --->
def detect(model, image_path=None, video_path=None):
    assert image_path or video_path

    # Image or video?
    # [1] --- image ---
    if image_path:
        # Run model detection and generate the color splash effect
        print("Running on {}".format(args.image))
        # Read image

        # accept folder (i.e. more than 1 image file)
        dataset_path = args.image
        images = os.listdir(dataset_path)
        images = [os.path.join(dataset_path, img) for img in images if img.split(".")[1] == "jpg"]
        for args.image in images:
            image = skimage.io.imread(args.image)
            # Detect objects
            r = model.detect([image], verbose=1)[0]           
            # Visualise objects
            img = apply_frame_notations(image=image
                  ,masks=r['masks']
                  ,boxes=r['rois']
                  ,centroids=[(int((x + a) / 2.0), int((y + b) / 2.0)) for y, x, b, a in r['rois']]
                  ,class_ids=r['class_ids']
                  ,scores=r['scores']
            )
            # Save 
            file_name = "bbox_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
            skimage.io.imsave(file_name, img)
    
    # [2] --- video ---
    elif video_path:
        # Video capture
        vcapture = cv2.VideoCapture(video_path)
        width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = vcapture.get(cv2.CAP_PROP_FPS)

        # v1.004 --->
        # Define codec and create video writer
        # file_name = "tracked_{:%Y%m%dT%H%M%S}.avi".format(datetime.datetime.now())
        # vwriter = cv2.VideoWriter(file_name,
        #                           cv2.VideoWriter_fourcc(*'MJPG'),
        #                           fps, (width, height))
        file_name = "tracked_{:%Y%m%dT%H%M%S}.mp4".format(datetime.datetime.now())
        vwriter = cv2.VideoWriter(file_name,
                                  cv2.VideoWriter_fourcc(*'mp4v'),
                                  fps, (width, height))
        # <--- v1.004

        # initialize object tracking
        # v1.004.3 --->
        if args.tracking == "Y":
            ct = CentroidTracker() # v1.002
        # <--- v1.004.3
        
        # <--- v1.002 
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
                mdl_centroids = [(int((x + a) / 2.0), int((y + b) / 2.0)) for y, x, b, a in r['rois']]

                # v1.002 --->
                # Object tracking
                if args.tracking == "Y": # v1.004.3
                    objects = ct.update(mdl_centroids)
                    objects = ct.verify(frame_memory=5, pass_rate=50)

                    # Delete objects predicted by model, but excluded based 
                    # on object tracking rules (suppress false positives)
                    obj_centroids = [tuple(v) for k, v in objects.items()]

                    # v1.004.2 --->
                    centroids = mdl_centroids.copy()
                    for mdl_obj in centroids:
                        if mdl_obj not in obj_centroids:
                            i = mdl_centroids.index(mdl_obj)
                            mdl_centroids.pop(i)
                            r['masks'] = np_pop(r['masks'], i)
                            r['rois'] = np_pop(r['rois'], i)
                            r['class_ids'] = np_pop(r['class_ids'], i)
                            r['scores'] = np_pop(r['scores'], i)
                    # <--- v1.004.2

                    # Display object id's 
                    for (objectID, (cX, cY)) in objects.items():
                        text = "ID: {}".format(objectID)
                        image = apply_cv_text(image, text, cX - 10, cY - 10)
                    # <--- v1.002

                # v1.008.0 --->
                # Keep record of detected objects...
                # frame, timestamp : MaxN
                MaxN = len(r['rois']) or 0
                # Display MaxN on frame
                image = apply_cv_text(image, "MaxN: {}".format(MaxN), 20, 20)

                # <--- v1.008.0

                # Visualise detected objects
                img = apply_frame_notations(image=image
                      ,masks=r['masks']
                      ,boxes=r['rois']
                      ,centroids=mdl_centroids
                      ,class_ids=r['class_ids']
                      ,scores=r['scores']
                )

                # Add image to video writer
                # OpenCV (RGB -> BGR)
                img = img[..., ::-1]
                vwriter.write(img)
                count += 1

        vwriter.release()
    print("Saved to ", file_name)
# <--- v1.001/2/4


############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect target fish species.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'splash' or 'detect'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/fish/dataset/",
                        help='Directory of the Fish dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image directory to apply color splash/ detection') # v1.003
    parser.add_argument('--video', required=False,
                        metavar="path or URL to video",
                        help='Video file to apply color splash/ detection') # v1.003
    parser.add_argument('--tracking', required=False,
                        metavar="include object tracking (Y/N)",
                        help='Y/N') # v1.004.3
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    # v1.003 --->
    elif args.command == "detect":
        assert args.image or args.video,\
               "Provide --image (directory) or --video (file) for model inference"
    # <--- v1.003

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = FishConfig()
    else:
        # class InferenceConfig(FishConfig):
        #     # Set batch size to 1 since we'll be running inference on
        #     # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
        #     GPU_COUNT = 1
        #     IMAGES_PER_GPU = 1
        config = FishInferenceConfig() # HERE_A
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
        weights_path = model.find_last()
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
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model)
    elif args.command == "splash":
        detect_and_color_splash(model, image_path=args.image,
                                video_path=args.video)
    # v1.001 --->
    elif args.command == "detect":
        detect(model, image_path=args.image,
                                video_path=args.video)
    # <--- v1.001
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'detect'".format(args.command))