#!/usr/bin/env python3

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo

def unregister_if_registered(name):
    """
    Purpose: Unregister the dataset if it is already registered.    
    :param name: The name of the dataset
    """
    if name in DatasetCatalog.list():
        DatasetCatalog.remove(name)
    if name in MetadataCatalog.list():
        MetadataCatalog.remove(name)

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    Purpose: Plot the confusion matrix
    :param cm: The confusion matrix
    :param classes: The class names
    :param normalize: Whether to normalize the confusion matrix
    :param title: The title of the confusion matrix
    :param cmap: The color map
    """
    # Here we create the confusion matrix plot
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    # Here we normalize the confusion matrix so match the YOLO confusion matrix 
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Here we format the confusion matrix plot
    thresh = cm.max() / 2.
    # The for loop will add the text to the confusion matrix plot
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, f"{cm[i, j]:.2f}",
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    
    # Here we format the confusion matrix plot for labels
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Main Function to visualize the confusion matrix
if __name__ == "__main__":
    dataset_names = ["my_dataset_test"]
    for name in dataset_names:
        unregister_if_registered(name)

    # Here we define the paths to the test annotations and images
    base_path = '/content/drive/MyDrive/CV_Project_2/Data/Detectron2'
    test_annotations_path = os.path.join(base_path, 'annotations/test_annotation.json')
    test_images_dir = os.path.join(base_path, 'images/test')

    # We can use the register_coco_instances function to register the test dataset
    register_coco_instances("my_dataset_test", {}, test_annotations_path, test_images_dir)
    # Here we define the configuration for the model
    cfg = get_cfg()
    # Garner basic architecture and weights for the model
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = os.path.join(base_path, "Runs/Final_Detectron_2/Test", "model_final.pth")
    # Here we need to define the number of classes and IoU threshold
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    # Here we create the predictor object
    predictor = DefaultPredictor(cfg)
    # Here we get the test dataset and background classes
    test_dataset_dicts = DatasetCatalog.get("my_dataset_test")
    thing_classes = MetadataCatalog.get("my_dataset_test").thing_classes + ['background']

    # Here we create the ground truth and predicted classes
    gt_classes = []
    pred_classes = []

    # For loop to get the ground truth and predicted classes
    for d in test_dataset_dicts:
        # Here we get the current image
        im = cv2.imread(d["file_name"])
        # Here we use the predictor object to get the predictions
        outputs = predictor(im)
        # Here we get the predicted classes
        output_classes = outputs["instances"].pred_classes.cpu().numpy()

        if len(output_classes) == 0:
            for ann in d['annotations']:
                gt_classes.append(ann['category_id'])
                pred_classes.append(0)
        else:
            for ann in d['annotations']:
                gt_classes.append(ann['category_id'])
                idxs = np.where(output_classes == ann['category_id'])[0]
                if len(idxs) > 0:
                    pred_classes.append(output_classes[idxs[0]])
                else:
                    pred_classes.append(0)

    # Here we create the confusion matrix
    cm = confusion_matrix(gt_classes, pred_classes, labels=range(len(thing_classes)))
    # Here we plot the confusion matrix that is normalized
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_normalized = np.nan_to_num(cm_normalized)
    # Here we plot the confusion matrix
    plt.figure(figsize=(10,10))
    plot_confusion_matrix(cm_normalized, classes=thing_classes, normalize=False, title='Normalized Confusion Matrix')
    plt.show()
