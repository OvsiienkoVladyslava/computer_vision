from enum import Enum

from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights,\
    RetinaNet_ResNet50_FPN_Weights


class DetectionModelsWeights(Enum):
    FASTER_RCNN = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    RETINANET = RetinaNet_ResNet50_FPN_Weights.DEFAULT
