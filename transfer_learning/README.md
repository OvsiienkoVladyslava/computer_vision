# Transfer learning for classification
### Goal
Use pretrained models 
CIFAR 10 source: https://www.cs.toronto.edu/~kriz/cifar.html
### Models performance

### How to predict classes of your data
1.Clone the repository
    
`git clone --recurse-submodules https://github.com/OvsiienkoVladyslava/computer_vision.git`  

2.Make sure that you fulfill all the requirements: Python 3.10 and other in requirements.txt.
 
`pip install -r requirements.txt`

3. Run in cmd or your IDE with parameters:  

--source-folder - path to folder with images, e.g. "D:/project/test_data"
--model-name - model name to use, available: 'Faster R-CNN, RetinaNet'
--score-threshold - min confidence threshold of prediction, e.g 0.8
--save-folder-path - path to folder where to save predicted labels, boxes, scores, images, e.g. "./test_results"
