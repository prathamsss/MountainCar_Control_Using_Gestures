# MountainCarV0 (Open Gym Environment) Control Using Hand Gestures

In this project we have controlled the mountain car from Open Gym Environment using Hand Gestures Recognition.
This was been acheived using MobileNetV2.
Following Actions belongs to:
                          - 0: Accelerate to the Left.
                          - 1: Don't accelerate.
                          - 2: Accelerate to the Right.
# Installation-  Clone this repository
      pip install -r /path/to/requirements.txt


# To Run this project, launch main.py file:
main.py - Contains script that runs inference on Real time Webcam and controls actions of MountainCarV0 from Open Gym.

        python main.py <model_file_path.pt>

          
Data_capture.py - Contains Script to record your own dataset. You can run this script by- 
        
        python Data_capture.py --data_dir <path/to/save/dataset> --mode <mode_train_test_valid>

To Train your own model, set hyperparameters in model_config.yml file and run Traning.py along with required arguments.
training.py - Pytorch Traning pipline.

        python training.py <path/to/model_config.yml>

Evaluation.py - Model Evaluation on costum dataset.
  
        python Evaluation.py --model_path <path/to/trained_model.pth> --dataset_path <path/to/dataset/for/evaluation>
        
Predict.py - Ulitity to Predict class on given image.

        python Predict.py --model_path <path/to/trained_model.pth> -- img_path <path/to/img>
        
model_conversion.py - Converts Pytorch state dictionary with Model instance to JIT Formate, for production use.
Utils.py - Utility to manage datasets.

       
       
Above are the instructions to launch this project.
