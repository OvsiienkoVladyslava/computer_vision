from enum import Enum

import torchvision
from torchvision.models import WeightsEnum
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_V2_Weights,
    RetinaNet_ResNet50_FPN_V2_Weights,
    fasterrcnn_resnet50_fpn_v2,
    retinanet_resnet50_fpn_v2,
)

from pytorch_object_detection.detection_models import DetectionPipeline


class DetectionModelsWeights(Enum):
    """
    Class of listed pre-trained weights for implemented object detection models.
    """

    FASTER_RCNN = FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1
    RETINANET = RetinaNet_ResNet50_FPN_V2_Weights.COCO_V1


class FasterRCNNDetection(DetectionPipeline):
    """
    Pretrained Faster R-CNN for object detection.
    """

    def load_weights(self) -> WeightsEnum:
        return DetectionModelsWeights.FASTER_RCNN.value

    def load_model(self) -> torchvision.models.detection:
        return fasterrcnn_resnet50_fpn_v2(weights=self.weights)


class RetinaNetDetection(DetectionPipeline):
    """
    Pretrained RetinaNet for object detection.
    """

    def load_weights(self) -> WeightsEnum:
        return DetectionModelsWeights.RETINANET.value

    def load_model(self) -> torchvision.models.detection:
        return retinanet_resnet50_fpn_v2(weights=self.weights)
