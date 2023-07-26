from enum import Enum

from pytorch_object_detection.detection_models import DetectionPipeline
from pytorch_object_detection.detection_models.models import (
    FasterRCNNDetection,
    RetinaNetDetection,
)


class DetectionModelsNames(Enum):
    RETINANET = RetinaNetDetection
    FASTER_RCNN = FasterRCNNDetection

    @classmethod
    def from_string(cls, model_name: str) -> DetectionPipeline:
        """
        Get model class by string name
        :param model_name: string name of detection model
        :return: model class
        """
        model_name = model_name.replace(" ", "_").lower()
        for model_type in cls:
            if model_type.name.lower() == model_name:
                return model_type.value
        raise AttributeError("This model is not implemented")
