from enum import Enum

from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_V2_Weights,
    RetinaNet_ResNet50_FPN_V2_Weights,
    fasterrcnn_resnet50_fpn_v2,
    retinanet_resnet50_fpn_v2,
)

from pytorch_object_detection.detection_models import DetectionPipeline


class DetectionModelsWeights(Enum):
    FASTER_RCNN = FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1
    RETINANET = RetinaNet_ResNet50_FPN_V2_Weights.COCO_V1


class FasterRCNNDetection(DetectionPipeline):
    """
    Pretrained Faster R-CNN for object detection.
    """

    def __init__(self):
        weights = DetectionModelsWeights.FASTER_RCNN.value
        model = fasterrcnn_resnet50_fpn_v2(weights=weights)
        super().__init__(weights=weights, model=model)


class RetinaNetDetection(DetectionPipeline):
    """
    Pretrained RetinaNet for object detection.
    """

    def __init__(self):
        weights = DetectionModelsWeights.RETINANET.value
        model = retinanet_resnet50_fpn_v2(weights=weights)
        super().__init__(weights=weights, model=model)
