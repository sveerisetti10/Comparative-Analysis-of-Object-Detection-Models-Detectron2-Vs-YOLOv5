#!/usr/bin/env python3

# Import necessary libraries
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultPredictor
import random
import cv2
import os
from detectron2.config import get_cfg

def visualize_predictions(dataset_name, predictor, n=5, output_dir=None):
    """
    Purpose: Visualize the predictions on the test dataset
    :param dataset_name: The name of the dataset    
    :param predictor: The predictor object
    :param n: The number of images to visualize
    :param output_dir: The output directory for the visualization images
    """
    # Here we get the dataset and metadata from the DatasetCatalog and MetadataCatalog
    dataset_dicts = DatasetCatalog.get(dataset_name)
    metadata = MetadataCatalog.get(dataset_name)

    # Here we create a for loop to visualize the predictions on the test dataset
    for i, d in enumerate(random.sample(dataset_dicts, n)):
        # This is the current image that we are visualizing
        img = cv2.imread(d["file_name"])
        # The predictor object will make the predictions on the current image
        outputs = predictor(img)
        # Here we use the Visualizer class to draw the predictions on the image
        v = Visualizer(img[:, :, ::-1], metadata=metadata, scale=0.5, instance_mode=ColorMode.IMAGE)
        # Here we get the output image from the Visualizer class
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        # Here we convert the output image to BGR format
        vis_img = out.get_image()[:, :, ::-1]

        # If there is an output directory, we will save the visualization images to the output directory
        if output_dir:
            # Create a 
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"vis_output_{i}.jpg")
            cv2.imwrite(output_path, vis_img)

# Main function
if __name__ == "__main__":
    # Here we gather the configuration and weights for the model
    cfg = get_cfg()
    # This is the path to the configuration file
    cfg.merge_from_file("/path/to/your/config/file.yaml")
    # These are the saved weights for the model
    cfg.MODEL.WEIGHTS = "/content/drive/MyDrive/CV_Project_2/Runs/Final_Detectron_2/Test/model_final.pth"
    # This is the predictor object
    predictor = DefaultPredictor(cfg)
    # Set the output directory for visualization images
    vis_output_dir = "/content/drive/MyDrive/CV_Project_2/Runs/Final_Detectron_2/Test/visualization_outputs/"
    # Visualize predictions on the test dataset
    visualize_predictions("my_dataset_test", predictor, n=5, output_dir=vis_output_dir)
