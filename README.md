## Instruction
    This project introduced the basics of the Open X-Embodiment dataset 
	and learned how to download the dataset in the appropriate data format. 
	Python files are organized according to official jupyter: loading datasets, visualization datasets, model inference.
    Give it a star if it helps you, thank you ~
    
## Video Explanations
- [Dataset Download](https://www.bilibili.com/video/BV1YG411C7XD/?spm_id_from=333.1007.top_right_bar_window_custom_collection.content.click&vd_source=63cd8055657905c0ac8a9388d7a972ed)
- [Code Performance](https://www.bilibili.com/video/BV1me411Q7rM/?spm_id_from=333.788&vd_source=63cd8055657905c0ac8a9388d7a972ed)

## Explanation of observation space、action space
    The model takes as input a RGB image from the robot workspace camera 
	and a task string describing the task that the robot is supposed to perform.
    The action dimensions we consider include seven variables for the gripper movement (x, y, z,roll, pitch, yaw, opening of the gripper). 
	Each variable represents the absolute value, the delta change to the dimension value or the velocity of the dimension.

## Code example
1. Install
    Create your code runtime environment and install official requirements libraries such as tensorflow, numpy, etc.
	For details, see Minimal_example_for_running_inference_using_RT_1_X_TF_using_tensorflow_datasets.ipynb.
```
    pip install rlds tf_agents dm-reverb[tensorflow] apache_beam tfp-nightly    
```
2. Download dataset
    get_dataset.txt provides the dataset download link. Please Watch a Video explanation
    I have also provided a download link to the Baidu web disk for part of the dataset in get_dataset.txt.
```
    (RT-X) PS G:\Open-X-Embodiment>gsutil  cp -n -r gs://gresearch/robotics/imperialcollege_sawyer_wrist_cam/0.1.0 E:\imperialcollege_sawyer_wrist_cam
```
3. Load dataset： 
    Pay attention to modify --> dataset_dir = '/home/communalien/Open-X-Embodiment/dataset/{}/0.1.0'.format(dataset_name) 
```
    python my_data_loader.py
```
4. Visual dataset and Inference:
```
    python my_visual_dataset.py
```
```
    python my_inference.py
```
## Reference
Official link: https://robotics-transformer-x.github.io./
