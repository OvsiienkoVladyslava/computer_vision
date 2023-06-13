from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights,\
    RetinaNet_ResNet50_FPN_Weights


class DetectionModelsWeights:
    RESNET = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    RETINANET = RetinaNet_ResNet50_FPN_Weights.DEFAULT