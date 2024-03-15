# Comparative Analysis of Object Detection Models: Identifying Drinks, Utensils, and Laptops Using Faster R-CNN and YOLO Architectures

This study provides a comparative analysis of two object detection frameworks: Detectron2, which uses a Faster R-CNN model architecture, and YOLOv5. Data augmentation and hyperparameter tuning were utilized to assess the performance of these models for object detection of common items such as drinks, utensils, and laptops. The evaluation criteria included mean Average Precision (mAP) at IoU thresholds of 0.50 (mAP@50) and averaged over IoU thresholds from 0.50 to 0.95 (mAP@50-95), alongside a deep-dive into the differences of the models’ inference speed and parameter size. Based on the results, it was concluded that YOLOv5 outperforms Detectron2 in performance on the test set, achieving mAP scores of 0.885 at mAP@50 and 0.656 at mAP@50-95, showing its better accuracy and generalizability across the three classes. Further, the YOLOv5 model demonstrated better efficiency with a shorter training duration and a larger model size, which did not hinder the performance, but rather allowed the model to explore more intricate and nuanced patterns within the images. It is proposed that further inclusion of a broader and more diverse dataset, coupled with more precise, accurate, and automated annotation techniques may enhance the robustness and accuracy of the model.

# Data Management
Data/
This directory includes a Python script, S3_Data_Download.py, which is used to interface with an AWS S3 bucket. The S3 bucket hosts a large collection of images and corresponding annotations necessary for training our object detection models. Utilizing external storage was necessary due to the substantial size of the dataset, which could not be efficiently hosted on GitHub.

To use this script:

Ensure AWS CLI is installed and configured with your credentials.
Run the S3_Data_Download.py script to fetch the dataset into your local environment.

# Model Development and Training
Notebooks/
The Jupyter notebooks within this directory detail the complete lifecycle of our object detection models, from inception to inference.

1. CV_Detectron2_Project2.ipynb:

Contains the process for developing the Detectron2 model, including hyperparameter tuning.
Provides the steps for training the model on the training set and validating on the validation set.
Demonstrates how to perform inferences on the test set.

2. CV_YOLOv5_Project2.ipynb:

Similar to the Detectron2 notebook, this file is dedicated to the YOLOv5 model's lifecycle.
It includes steps for model development, hyperparameter tuning, training, and validation.
Guides through the inference process on the test set.

Scripts/
Scripts in this directory allow for automated, script-based interaction with the models, mirroring the processes laid out in the Jupyter notebooks.

For instance:
- Detectron2_Train.py: Train the Detectron2 model.
- Detectron2_Test.py: Perform inference using the trained Detectron2 model.
- Detectron2_Confusion_Matrix.py: Generate a confusion matrix from the inference results of the Detectron2 model.

To execute a script:
1. Navigate to the Scripts/ directory in your terminal.
2. Run the desired script with appropriate command-line arguments (if any).

# Understanding Results
The notebooks and scripts facilitate a comprehensive understanding of model performance and behavior. For advanced analysis, such as creating confusion matrices or other metrics, refer to the corresponding scripts mentioned above.

