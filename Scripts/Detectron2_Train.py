#!/usr/bin/env python3

# Import necessary libraries
from detectron2.data import detection_utils as utils
import copy
import torch
from detectron2.data.transforms import (
    apply_transform_gens,
    RandomFlip,
    RandomRotation,
    ResizeShortestEdge,
    RandomBrightness,
    RandomContrast,
    RandomSaturation,
    RandomCrop,
    RandomLighting,  
    RandomExtent  
)
from detectron2.structures import BoxMode
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.data import build_detection_train_loader, build_detection_test_loader
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data.datasets import register_coco_instances
import os

def custom_mapper(dataset_dict):
    """
    Purpose: Custom mapper function to apply transformations to the dataset
    Input: dataset_dict (dict) - dictionary containing the dataset
    """
    dataset_dict = copy.deepcopy(dataset_dict)
    # Here we will apply the following transformations to the dataset
    image = utils.read_image(dataset_dict["file_name"], format="BGR")
    transform_list = [
        RandomFlip(prob=0.5, horizontal=True, vertical=False),
        RandomRotation(angle=[-10, 10], expand=False, sample_style='range'),
        ResizeShortestEdge(short_edge_length=[640, 672, 704, 736, 768, 800], max_size=1333, sample_style='choice'),
        RandomBrightness(0.8, 1.2),
        RandomContrast(0.8, 1.2),
        RandomSaturation(0.8, 1.2),
        RandomCrop("relative_range", (0.8, 0.8)),
        RandomLighting(0.8),
        RandomExtent(scale_range=(0.8, 1.2), shift_range=(0.0, 0.2))
    ]
    image, transforms = apply_transform_gens(transform_list, image)
    dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))
    # If the dataset contains annotations, we will apply the transformations to the annotations as well
    if "annotations" in dataset_dict:
        # Here we will apply the following transformations to the annotations
        annos = [
            utils.transform_instance_annotations(obj, transforms, image.shape[:2])
            for obj in dataset_dict.pop("annotations")
        ]
        # Here we make sure that the bounding box format is in XYWH_ABS
        for anno in annos:
            anno["bbox"] = BoxMode.convert(anno["bbox"], BoxMode.XYXY_ABS, BoxMode.XYWH_ABS)
        dataset_dict["annotations"] = annos
        # Return the transformed dataset
    return dataset_dict

# The purpose of the class is to create a custom trainer that will use the custom_mapper function to apply transformations to the dataset
class CustomTrainer(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=custom_mapper)

# To avoid running into an error with the dataset already being registered, we will unregister the dataset if it is already registered
def unregister_dataset(name):
    """
    Purpose: Unregister the dataset if it is already registered
    Input: name (str) - name of the dataset
    """
    from detectron2.data import DatasetCatalog, MetadataCatalog
    if name in DatasetCatalog.list():
        DatasetCatalog.remove(name)
    if name in MetadataCatalog.list():
        MetadataCatalog.remove(name)

# Main function
if __name__ == "__main__":
    # Here we will define the paths to the images and annotations
    base_path = '/content/drive/MyDrive/CV_Project_2/Data/Detectron2'
    train_annotations_path = f'{base_path}/annotations/train_annotations.json'
    val_annotations_path = f'{base_path}/annotations/val_annotations.json'
    test_annotations_path = f'{base_path}/annotations/test_annotation.json'
    train_images_dir = f'{base_path}/images/train'
    val_images_dir = f'{base_path}/images/val'
    test_images_dir = f'{base_path}/images/test'

    # Here we create unique names for the datasets
    dataset_names = ["my_dataset_train", "my_dataset_val", "my_dataset_test"]
    for name in dataset_names:
        unregister_dataset(name)

    # Here we register the datasets
    register_coco_instances("my_dataset_train", {}, train_annotations_path, train_images_dir)
    register_coco_instances("my_dataset_val", {}, val_annotations_path, val_images_dir)
    register_coco_instances("my_dataset_test", {}, test_annotations_path, test_images_dir)

    # Here we will define the configuration for the model
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("my_dataset_train",)
    cfg.DATASETS.TEST = ("my_dataset_val",)
    cfg.DATALOADER.NUM_WORKERS = 4
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")  
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.0025  
    cfg.SOLVER.MAX_ITER = 12000   
    cfg.SOLVER.GAMMA = 0.1
    cfg.SOLVER.STEPS = (3000, 6000)  
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3  
    cfg.OUTPUT_DIR = '/content/drive/MyDrive/CV_Project_2/Runs/Final_Detectron_2/Validation'

    # If there is no output directory, we will create one
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    # Here we will create the custom trainer and train the model
    trainer = CustomTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

    # Here we use the COCOEvaluator to evaluate the model
    evaluator = COCOEvaluator("my_dataset_val", cfg, False, output_dir=cfg.OUTPUT_DIR)
    # We load in the validation dataset
    val_loader = build_detection_test_loader(cfg, "my_dataset_val")
    # We perform inference on the validation dataset
    evaluation_results = inference_on_dataset(trainer.model, val_loader, evaluator)
    # 
    print(evaluation_results)
