# Object Detection
### Goal
Use pretrained models from pytorch for object detection.
Source: https://pytorch.org/vision/stable/models.html
### Models performance
- _Faster R-CNN_ - improved Faster R-CNN model with a ResNet-50-FPN backbone from 'Benchmarking Detection Transfer Learning with Vision Transformers' paper use weights pretrained on COCO dataset, box mAP = 46.7 on COCO-val2017
- _RetinaNet_ - improved RetinaNet model with a ResNet-50-FPN backbone, use weights pretrained on COCO dataset, box mAP = 36.4 on COCO-val2017

### How to predict classes of your data
1. Clone the repository
    
`git clone --recurse-submodules https://github.com/OvsiienkoVladyslava/computer_vision.git`  

2. Make sure that you fulfill all the requirements: Python 3.10 and other in requirements.txt.
 
`pip install -r requirements.txt`

If you want to run your code using GPU you need to install [cuda version 11.7](https://developer.nvidia.com/cuda-11-7-0-download-archive?target_os=Windows&target_arch=x86_64&target_version=11) (to check if you've installed it run in cmd `nvcc --version`)

3. Run in cmd or your IDE with parameters:  

_--source-folder_ - path to folder with images, e.g. "D:\\project\\test_data"<br />
_--model-name_ - model name to use, available: 'Faster R-CNN, RetinaNet'<br />
_--score-threshold_ - min confidence threshold of prediction, e.g 0.8<br />
_--save-folder-path_ - path to folder where to save predicted labels, boxes, scores, images, e.g. "./test_results"<br />
_--run_on_gpu_ - flag, if present - run code on GPU if cuda is available otherwise run on CPU

4. Example

**Console input:** <br />
`$ python run.py --source-folder "D:\\computer vision\\pytorch_object_detection\\test_data" --model-name "RetinaNet" --score-threshold 0.8 --save-folder-path "./test_results" --run_on_gpu`<br />
There is only 1 image (object_detection_example.jpg) in /test_data 

**Image result:** <br />
![Detection result](..\README_images\object_detection_example.jpg)

**JSON file result:**<br />
>{ <br />
"labels": [["dog", "dog", "dog", "bird"]],<br />
"boxes": [[...]],<br />
"scores": [[0.8993059396743774, 0.8599454760551453, 0.8489809036254883, 0.8322062492370605]],<br />
"images": ["object_detection_example.jpg"] <br />
}