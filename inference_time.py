from multiprocessing import freeze_support

import torch

# assert torch.__version__.startswith("1.8")
import torchvision
import cv2
import os
import numpy as np
import json
import json
from scipy.spatial.distance import euclidean
from imutils import perspective
from imutils import contours
import imutils
import cv2
import time
import werkzeug
from PIL import Image
from PIL.ExifTags import TAGS
import datetime
import multiprocessing

import albumentations as A
import matplotlib.pyplot as plt
from detectron2.data.transforms import ResizeShortestEdge, RandomFlip
from detectron2.structures import BoxMode
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset


# classes = ['stem', 'pumpkin_fruit', 'tree_stem']


def main():
    # microcontroller_metadata = MetadataCatalog.get("category_train")
    print(torch.cuda.is_available())
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    #cfg.DATASETS.TRAIN = ("category_train",)
    #cfg.DATASETS.TEST = ()
    #cfg.DATALOADER.NUM_WORKERS = 10
    #cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    #cfg.SOLVER.IMS_PER_BATCH = 2
    #cfg.SOLVER.BASE_LR = 0.00025
    #cfg.SOLVER.MAX_ITER = 1
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3

    # os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    # trainer = DefaultTrainer(cfg)
    # trainer.resume_or_load(resume=False)
    # trainer.train()
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.50
    cfg.DATASETS.TEST = ("skin_test",)
    # class_names = ["stem", "pumpkin_fruit", 'tree_stem']
    # score_threshs = [0.1, 0.8, 0.1]  # Example score thresholds for each class
    # score_thresh_mapping = {class_names[i]: score_threshs[i] for i in range(len(class_names))}
    predictor = DefaultPredictor(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predictor.model.to(device)
    print(predictor)

    # test_dataset_dicts = get_data_dicts(data_path + 'test', classes)
    # img = cv2.imread('H:\\Agriculture Project\\Pumpkin\\Pumpkin Dataset\\Pumpkin Dataset\\365 (1).JPG')
    # img_path = 'H:\\Agriculture Project\\Pumpkin\\Pumpkin Dataset\\Pumpkin Dataset'

    img1 = cv2.imread("H:\\Agriculture Project\\Pumpkin\\Pumpkin Dataset\\checking_inference\\389 (4).JPG")

    a = time.time()
    print(a)

    # Process the image
    outputs = predictor(img1)
    instances = outputs["instances"].to("cuda")
    print(instances)
    b = time.time()
    print(b)
    sec = b - a
    print(sec)
    fps = 1.0 / sec

    # Calculate FPS every 10 frames
    print("Frame Per Second (FPS):", fps)

    """
    instances = outputs["instances"].to("cpu")

    masks = instances.pred_masks  # Get the segmentation masks of instances

    for i in range(len(masks)):
        # Create a binary mask for the instance
        mask = masks[i].cpu().numpy()
        mask = np.uint8(mask) * 255
        class_id = instances.pred_classes[i].item()  # Get the class ID of the instance
        class_name = microcontroller_metadata.get("thing_classes")[class_id]  # Get the class name of the instance
        # Save the mask as an image
        cv2.imwrite("mask_{}.png".format(i), mask)
    for i in range(len(instances)):  # Iterate over each instance
        class_id = instances.pred_classes[i].item()  # Get the class ID of the instance
        class_name = microcontroller_metadata.get("thing_classes")[class_id]  # Get the class name of the instance
        # Convert the binary mask to a numpy array and find the non-zero coordinates
        mask = masks[i].numpy()

        nonzero_coords = np.nonzero(mask)

        # Calculate the width and height of the instance
        xmin = np.min(nonzero_coords[1])
        xmax = np.max(nonzero_coords[1])
        ymin = np.min(nonzero_coords[0])
        ymax = np.max(nonzero_coords[0])
        width = xmax - xmin
        height = ymax - ymin
        print("width in pixles", width)
        print("height in pixles", height)

        # Convert the width and height to millimeters
        width_mm = width / 42.12
        height_mm = height / 43.22

        print(f"Class: {class_name}, Width: {width_mm:.2f}, Height: {height_mm:.2f}")

        # Draw a rectangle around the instance with the calculated width and height
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        cv2.putText(img, f"Class: {class_name}, Width: {height_mm:.2f}, Height: {height_mm:.2f}",
                    (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    plt.figure(figsize=(60, 60))
    plt.imshow(cv2.cvtColor(img[:, :, ::-1], cv2.COLOR_BGR2RGB))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
"""


if __name__ == '__main__':
    main()
