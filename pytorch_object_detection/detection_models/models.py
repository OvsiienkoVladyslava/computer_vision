from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, retinanet_resnet50_fpn
from pytorch_object_detection.detection_models import DetectionPipeline
from pytorch_object_detection.detection_models.configs import DetectionModelsWeights


class FasterRCNNDetection(DetectionPipeline):
    """
    Pretrained Faster R-CNN for object detection.
    """
    def __init__(self):
        self.weights = DetectionModelsWeights.FASTER_RCNN
        self.model = fasterrcnn_resnet50_fpn_v2(
            weights=self.weights,
            box_score_thresh=0.9
        )
        super().__init__()


class RetinaNetDetection(DetectionPipeline):
    """
    Pretrained RetinaNet for object detection.
    """
    def __init__(self):
        self.weights = DetectionModelsWeights.RETINANET
        self.model = retinanet_resnet50_fpn(
            weights=self.weights,

        )
        super().__init__()