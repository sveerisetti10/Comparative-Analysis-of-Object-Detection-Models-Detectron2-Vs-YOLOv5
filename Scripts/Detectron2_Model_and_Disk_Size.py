#!/usr/bin/env python3

import os
import torch
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer

def get_model_parameters(model):
    """
    Purpose: Get the number of parameters in the model.
    Args:
        model: The model object.
    """
    # Here we use a list comprehension to get the number of parameters in the model
    return sum(p.numel() for p in model.parameters())

def get_disk_size_of_file(filepath):
    """
    Purpose: Get the disk size of the model file.
    Args:
        filepath: The file path of the model.
    """
    # Here we use the os.path.getsize function to get the disk size of the model file
    return os.path.getsize(filepath)

# Main function
if __name__ == "__main__":
    # Here we gather the configuration and weights for the model
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = os.path.join('/content/drive/MyDrive/CV_Project_2/Runs/Final_Detectron_2/Test', "model_final.pth")
    # Explicitly set the number of classes to 3
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3

    # Here we build the model
    model = build_model(cfg)
    # Here we load the weights of the model
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
    # Here we use the get_model_parameters function to get the number of parameters in the model
    num_parameters = get_model_parameters(model)
    print(f'Number of parameters in model: {num_parameters}')
    model_file_path = cfg.MODEL.WEIGHTS
    # Here we use the get_disk_size_of_file function to get the disk size of the model file
    disk_size = get_disk_size_of_file(model_file_path)
    # Print the disk size of the model file
    print(f'Disk size of the model file: {disk_size} bytes')
    # Alternatively, we can convert the disk size to MB
    print(f'Disk size of the model file: {disk_size / (1024*1024)} MB')
