import torchvision
import torch
from torch import Tensor
from torchvision.io.image import read_image
from torchvision.models import WeightsEnum

from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image

from abc import ABC
from typing import List


class DetectionPipeline(ABC):
    """
    Class defining basic functionality to run pretrained object detection model on image or batch of images.
    In __init__ func of inherit class it is needed to:
        1. define self.weights of pretrained model
        2. define self.model with these weights
        3. call init of this class
    """
    model: torchvision.models.detection
    weights: WeightsEnum
    preprocess_pipeline: torchvision.transforms._presets.ObjectDetection
    model_classes: list

    def __init__(self):
        self.model.eval()
        self.model_classes = self.weights.meta["categories"]
        self.preprocess_pipeline = self.weights.transforms()

    def _image_preprocess(self, image_paths: List[str] | str) -> (List[Tensor], List[Tensor]):
        """
        Read and preprocesses images for the model
        :param image_paths: path's to images or image
        :return: read images and preprocessed for detection model
        """
        if type(image_paths) is str:
            image_paths = [image_paths]

        raw_images = [read_image(path) for path in image_paths]
        batch = [self.preprocess_pipeline(img) for img in raw_images]

        return raw_images, batch

    @staticmethod
    def visualize_result(imgs: List[Tensor], boxes: List[list], labels: List[List[str]]) -> None:
        """
        Visualize detected objects - image with predicted boxes and labels
        :param imgs: image on which objects wer detected
        :param boxes: predicted boxes of objects
        :param labels: predicted labels of objects
        """
        for ind, img in enumerate(imgs):
            box = torch.stack(boxes[ind], dim=0)
            drawed_boxes = draw_bounding_boxes(img, boxes=box,
                                               labels=labels[ind],
                                               width=2)
            im = to_pil_image(drawed_boxes.detach())
            im.show(title=f'Image №{ind}')

    def _filter_detection_output(self, predictions: dict, threshold: float) -> (list, list, list):
        """
        Filter detected results - labels, boxes, confidence scores based on threshold of score - leave if <= threshold
        :param predictions: output of model
        :param threshold: threshold of confidence to filter results
        :return: filtered predicted labels, boxes and confidence scores
        """
        # Process and filter output of model
        output_labels, output_boxes, output_scores = [], [], []
        for output in predictions:
            labels, boxes, scores = [], [], []
            for ind, score in enumerate(output['scores']):
                if score >= threshold:
                    scores.append(score)
                    labels.append(self.model_classes[output["labels"][ind]])
                    boxes.append(output['boxes'][ind])
            output_scores.append(scores)
            output_labels.append(labels)
            output_boxes.append(boxes)

        return output_labels, output_boxes, output_scores

    def run(self, image_paths: List[str] | str, score_threshold: float = 0.9) -> (list, list, list):
        """
        Run predictions of the model.
        :param image_paths: path's to images or image on which detect objects
        :param score_threshold: threshold of confidence to filter results (leave if <= threshold)
        :return: predicted labels, boxes and confidence scores
        """
        # Read and preprocess images
        raw_images, batch = self._image_preprocess(image_paths)

        # Get output of model
        outputs = self.model(batch)

        # Process and filter output of model
        output_labels, output_boxes, output_scores = self._filter_detection_output(outputs, score_threshold)

        # Visualize detection results
        self.visualize_result(raw_images, output_boxes, output_labels)

        return output_labels, output_boxes, output_scores
