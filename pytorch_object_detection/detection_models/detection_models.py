from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, retinanet_resnet50_fpn
from detection_models import DetectionPipeline
from detection_models.configs import DetectionModelsWeights


class ResNetDetection(DetectionPipeline):
    def __init__(self):
        self.weights = DetectionModelsWeights.RESNET
        self.model = fasterrcnn_resnet50_fpn_v2(
            weights=self.weights,
            box_score_thresh=0.9
        )
        super().__init__()


class RetinaNetDetection(DetectionPipeline):
    def __init__(self):
        self.weights = DetectionModelsWeights.RETINANET
        self.model = retinanet_resnet50_fpn(
            weights=self.weights,

        )
        super().__init__()


if __name__ == '__main__':

    path = "../test_data/cat_duck_dog.jpg"

    model = ResNetDetection()
    labels, boxes, scores = model.run(path, score_threshold=0.5)

