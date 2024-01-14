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

classes = ['stem', 'pumpkin_fruit', 'tree_stem']

data_path = 'H:\\Agriculture Project\\new_labeled_data\\'

werkzeug.cached_property = werkzeug.utils.cached_property


def get_data_dicts(directory, classes):
    dataset_dicts = []
    for filename in [file for file in os.listdir(directory) if file.endswith('.json')]:
        json_file = os.path.join(directory, filename)
        with open(json_file) as f:
            img_anns = json.load(f)

        record = {}

        filename = os.path.join(directory, img_anns["imagePath"])
        record["file_name"] = filename
        record["height"] = img_anns["imageHeight"]
        record["width"] = img_anns["imageWidth"]
        # Load image using OpenCV
        image = cv2.imread(filename)
        # Convert image from BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Apply data augmentation
        transform = A.Compose([
            A.HorizontalFlip(p=0.5),  # Flip horizontally with 50% probability
            A.RandomBrightnessContrast(p=0.2),  # Adjust brightness and contrast randomly with 20% probability
            A.Rotate(limit=10, p=0.2),  # Rotate image by up to 10 degrees with 20% probability
            # Add more augmentations as desired
        ])
        augmented = transform(image=image)
        image = augmented["image"]

        # Update image size in the record after augmentation
        record["height"], record["width"], _ = image.shape

        # Resize image
        resize_shortest_edge = ResizeShortestEdge(800, 1333)
        # Get the transformation matrix and apply it to the image
        transform_matrix = resize_shortest_edge.get_transform(image).apply_image(image)

        # Update file name in the record after resizing
        record["file_name"] = filename

        annos = img_anns["shapes"]
        objs = []
        for anno in annos:
            px = [a[0] for a in anno['points']]  # x coord
            py = [a[1] for a in anno['points']]  # y-coord
            poly = [(x, y) for x, y in zip(px, py)]  # poly for segmentation
            poly = [p for x in poly for p in x]

            obj = {
                "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [poly],
                "category_id": classes.index(anno['label']),
                "iscrowd": 0
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts


def main():
    result_dict = {}  # Dictionary to store the measurement results
    for d in ["train", "test"]:
        DatasetCatalog.register(
            "category_" + d,
            lambda d=d: get_data_dicts(data_path + d, classes)
        )
        MetadataCatalog.get("category_" + d).set(thing_classes=classes)

    microcontroller_metadata = MetadataCatalog.get("category_train")

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("category_train",)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = 50000
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3

    # os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    # trainer = DefaultTrainer(cfg)
    # trainer.resume_or_load(resume=False)
    # trainer.train()
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.50
    cfg.DATASETS.TEST = ("skin_test",)
    class_names = ["stem", "pumpkin_fruit", 'tree_stem']
    # score_threshs = [0.1, 0.8, 0.1]  # Example score thresholds for each class
    # score_thresh_mapping = {class_names[i]: score_threshs[i] for i in range(len(class_names))}
    predictor = DefaultPredictor(cfg)

    # test_dataset_dicts = get_data_dicts(data_path + 'test', classes)
    # img = cv2.imread('H:\\Agriculture Project\\Pumpkin\\Pumpkin Dataset\\Pumpkin Dataset\\365 (1).JPG')
    img_path = 'H:\\Agriculture Project\\Pumpkin\\Pumpkin Dataset\\Pumpkin Dataset'
    output_list = []
    processed_images = set()  # Set to keep track of processed image names
    for filename in os.listdir(img_path):
        if filename.endswith(".JPG") or filename.endswith(".png"):
            img_name = filename
            if img_name in processed_images:
                continue  # Skip processing if the image name has already been processed
            processed_images.add(img_name)

            orginail_img = os.path.join(img_path, filename)
            image = Image.open(orginail_img)
            image_width = image.size[0]
            image_height = image.size[1]

            exif_data = image._getexif()

            for tag_id, value in exif_data.items():
                tag_name = TAGS.get(tag_id, tag_id)
                if tag_name == "DateTimeOriginal":
                    date_time_taken = value
                    break
            date_taken = datetime.datetime.strptime(date_time_taken, "%Y:%m:%d %H:%M:%S").date()

            img = cv2.imread(orginail_img)
            outputs = predictor(img)
            instances = outputs["instances"].to("cpu")
            masks = instances.pred_masks  # Get the segmentation masks of instances

            for i in range(len(masks)):
                # Create a binary mask for the instance
                mask = masks[i].cpu().numpy()
                mask = np.uint8(mask) * 255
                class_id = instances.pred_classes[i].item()  # Get the class ID of the instance
                class_name = microcontroller_metadata.get("thing_classes")[
                    class_id]  # Get the class name of the instance
                # Save the mask as an image
                cv2.imwrite("mask_{}.png".format(i), mask)

            image_data = {
                "image_name": img_name,
                "germplasm": img_name[:3],
                "capture_date": str(date_taken),
                "width": str(image_width),
                "height": str(image_height)

            }
            for i in range(len(instances)):  # Iterate over each instance
                class_id = instances.pred_classes[i].item()  # Get the class ID of the instance
                class_name = microcontroller_metadata.get("thing_classes")[
                    class_id]  # Get the class name of the instance
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

                width_mm = width / 42.12
                height_mm = height / 43.22
                # print("width in pixles", width)
                # print("height in pixles", height)

                instance_key_h = class_name + '_h'
                instance_key_w = class_name + '_w'

                image_data[instance_key_h] = '{:.2f}'.format(height_mm)
                image_data[instance_key_w] = '{:.2f}'.format(width_mm)
            output_list.append(image_data)
    print(output_list)
    output_json = json.dumps(output_list, indent=4)

    # Specify the file path to save the JSON file
    json_file_path = 'output.json'

    # Write the JSON data to the file
    with open(json_file_path, 'w') as json_file:
        json_file.write(output_json)
        # print(f"Class: {class_name}, Width: {width_mm:.2f}, Height: {height_mm:.2f}")
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    cv2.imwrite('gray.png', gray)
    # Apply Canny edge detection to detect edges
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Define the Hough parameters for detecting lines
    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180
    threshold = 100
    # min_line_length = int(0.75 * max(img.shape))  # minimum number of pixels making up a line
    min_line_length = int(0.75 * max(img.shape))
    max_line_gap = 300
    min_line_ppr = (max_line_gap / 70) * 100
    print(min_line_ppr)
    # Detect lines using the Hough Line Transform
    lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)
    cv2.imwrite('edge.png', lines)
    longest_line = None
    longest_line_length = 0
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if (x2 < 1000 and x1 > 90):
            line_length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            if line_length > longest_line_length:
                longest_line = line
                longest_line_length = line_length

    # Draw the longest line
    final_ratio = -1
    if longest_line is not None:
        x1, y1, x2, y2 = longest_line[0]
        if (y2 > y1):
            print(str((y2 - y1) / 70) + " pixels per mm")
            final_ratio = (y2 - y1) / 70
            print("real length is " + str(300 * (1 / ((y2 - y1) / 100))))
            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 20)
        else:
            print(str((y1 - y2) / 70) + " pixels per mm")
            final_ratio = (y1 - y2) / 70
            print("real length is " + str(300 * (1 / ((y1 - y2) / 100))))
            cv2.line(img, (x2, y2), (x1, y1), (0, 0, 255), 20)
    cv2.imwrite('detected_pumpkin.png', img)
    print(final_ratio)
    my_ratio_for_width = 46
    outputs = predictor(img)
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
