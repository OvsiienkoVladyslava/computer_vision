from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, retinanet_resnet50_fpn_v2
from pytorch_object_detection.detection_models import DetectionPipeline
from pytorch_object_detection.detection_models.configs import DetectionModelsWeights


class FasterRCNNDetection(DetectionPipeline):
    """
    Pretrained Faster R-CNN for object detection.
    """
    def __init__(self):
        weights = DetectionModelsWeights.FASTER_RCNN.value
        model = fasterrcnn_resnet50_fpn_v2(
            weights=self.weights
        )
        super().__init__(weights=weights, model=model)


class RetinaNetDetection(DetectionPipeline):
    """
    Pretrained RetinaNet for object detection.
    """
    def __init__(self):
        weights = DetectionModelsWeights.RETINANET.value
        model = retinanet_resnet50_fpn_v2(
            weights=self.weights
        )
        super().__init__(weights=weights, model=model)
