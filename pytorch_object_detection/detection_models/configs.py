from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights,\
    RetinaNet_ResNet50_FPN_V2_Weights


class DetectionModelsWeights:
    FASTER_RCNN = FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1
    RETINANET = RetinaNet_ResNet50_FPN_V2_Weights.COCO_V1
