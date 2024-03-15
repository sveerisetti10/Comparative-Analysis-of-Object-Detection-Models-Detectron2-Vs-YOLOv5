
import argparse
import yaml
import os

def parse_args():
    """
    Purpose: Parse command line arguments
    Input: None
    Please make sure to update the default values for the arguments as per your directory structure.
    """
    parser = argparse.ArgumentParser(description="Script to prepare dataset and hyperparameters, train, and validate a YOLO model.")
    parser.add_argument('--data_dir', type=str, default='/content/drive/MyDrive/CV_Project_2/Data/YOLO', help='Base directory for the dataset')
    parser.add_argument('--yolo_dir', type=str, default='/content/yolov5', help='Base directory for YOLO')
    parser.add_argument('--runs_dir', type=str, default='/content/drive/MyDrive/CV_Project_2/Runs', help='Base directory for runs')
    parser.add_argument('--weights', type=str, default='yolov5l.pt', help='Initial weights path')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs for training')
    parser.add_argument('--img_size', type=int, default=640, help='Image size for training and validation')
    parser.add_argument('--batch_size_val', type=int, default=32, help='Batch size for validation')
    args = parser.parse_args()
    return args

def setup_dataset(data_dir):
    """
    Purpose: Create a dataset.yaml file for the YOLO model
    Input: data_dir (str) - Base directory for the dataset
    """
    dataset_yaml = {
        'path': data_dir, 
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',  
        'nc': 3, 
        'names': ['drink', 'utensil', 'laptop']  
    }
    with open(os.path.join(data_dir, 'dataset.yaml'), 'w') as file:
        yaml.dump(dataset_yaml, file)

def modify_hyp(yolo_dir):
    """
    Purpose: Modify the hyperparameters for the YOLO model
    Input: yolo_dir (str) - Base directory for YOLO
    """
    existing_hyp_path = os.path.join(yolo_dir, 'data/hyps/hyp.scratch-low.yaml')
    with open(existing_hyp_path, 'r') as file:
        hyp = yaml.safe_load(file)
    # Modify hyperparameters as needed
    hyp['optimizer'] = 'adam'
    hyp['mosaic'] = 1.0  
    hyp['mixup'] = 0.2  
    hyp['jitter'] = 0.2 
    hyp['hsv_h'] = 0.015  
    hyp['hsv_s'] = 0.7  
    hyp['hsv_v'] = 0.4 
    hyp['degrees'] = 0.2  
    hyp['translate'] = 0.1 
    hyp['scale'] = 0.5 
    hyp['shear'] = 0.1 
    custom_hyp_path = os.path.join(yolo_dir, 'data/hyps/hyp.custom.yaml')
    with open(custom_hyp_path, 'w') as file:
        yaml.dump(hyp, file)
    return custom_hyp_path

def train_model(yolo_dir, data_dir, runs_dir, weights, batch_size, epochs, img_size, custom_hyp_path):
    """
    Purpose: Train the YOLO model
    Input: yolo_dir (str) - Base directory for YOLO
           data_dir (str) - Base directory for the dataset
           runs_dir (str) - Base directory for runs
           weights (str) - Initial weights path
           batch_size (int) - Batch size for training
           epochs (int) - Number of epochs for training
           img_size (int) - Image size for training and validation
           custom_hyp_path (str) - Path to the custom hyperparameters
    """
    project_dir = os.path.join(runs_dir, 'YOLO_Final/Validation')
    os.makedirs(project_dir, exist_ok=True)
    train_command = f"python {os.path.join(yolo_dir, 'train.py')} --img {img_size} --batch {batch_size} --epochs {epochs} --data {os.path.join(data_dir, 'dataset.yaml')} --weights {weights} --hyp {custom_hyp_path} --project {project_dir} --name YOLOv5_Final_Validation --exist-ok"
    os.system(train_command)

def validate_model(yolo_dir, data_dir, runs_dir, weights_path, img_size, batch_size_val):
    """
    Purpose: Validate the YOLO model
    Input: yolo_dir (str) - Base directory for YOLO
           data_dir (str) - Base directory for the dataset
           runs_dir (str) - Base directory for runs
           weights_path (str) - Path to the best model weights
           img_size (int) - Image size for training and validation
           batch_size_val (int) - Batch size for validation
    """
    project_dir = os.path.join(runs_dir, 'YOLO_Final/Test')
    os.makedirs(project_dir, exist_ok=True)
    val_command = f"python {os.path.join(yolo_dir, 'val.py')} --weights {weights_path} --data {os.path.join(data_dir, 'dataset.yaml')} --img {img_size} --batch {batch_size_val} --task test --project {project_dir} --name YOLOv5_Test --exist-ok"
    os.system(val_command)

# Main Function 
def main():
    """
    Purpose: Main function to prepare dataset and hyperparameters, train, and validate a YOLO model
    Here we combine the functions to prepare the dataset, modify the hyperparameters, train the model, and validate the model.
    """
    args = parse_args()
    setup_dataset(args.data_dir)
    custom_hyp_path = modify_hyp(args.yolo_dir)
    train_model(args.yolo_dir, args.data_dir, args.runs_dir, args.weights, args.batch_size, args.epochs, args.img_size, custom_hyp_path)
    # Once the training is complete, we can validate the model. Please make sure to update the weights path accordingly to your best model weights.
    weights_path = '/content/drive/MyDrive/CV_Project_2/Runs/Experiment_Full_New_YOLO_CVAT6/weights/best.pt' 
    validate_model(args.yolo_dir, args.data_dir, args.runs_dir, weights_path, args.img_size, args.batch_size_val)

# Run the main function
if __name__ == '__main__':
    main()
