import json
import os

import torchvision
import torch
from torch import Tensor
from torchvision.io.image import read_image
from torchvision.models import WeightsEnum

from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image

from typing import List


class DetectionPipeline:
    """
    Class defining basic functionality to run pretrained object detection model on image or batch of images.
    """

    def __init__(self, weights: WeightsEnum, model: torchvision.models.detection):
        """
        In __init__ func of child class it is needed to:
        1. define weights of pretrained model
        2. define model with these weights
        3. call init of this class

        :param weights: pre-trained weights of model from torchvision (they are specified in WeightsEnum class )
        :param model: initialized model with weights
        """
        self.weights = weights
        self.model = model
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
    def draw_boxes(imgs: List[Tensor], boxes: List[list], labels: List[List[str]]) -> list:
        """
        Visualize detected objects - image with predicted boxes and labels

        :param imgs: image on which objects wer detected
        :param boxes: predicted boxes of objects
        :param labels: predicted labels of objects
        """
        images_with_boxes = []
        for ind, img in enumerate(imgs):
            predicted_boxes = torch.Tensor(boxes[ind])
            try:
                drawn_boxes = draw_bounding_boxes(img, boxes=predicted_boxes,
                                                  labels=labels[ind],
                                                  width=2)
                images_with_boxes.append(to_pil_image(drawn_boxes.detach()))
            except IndexError:
                # Not found boxes for image so not draw them
                images_with_boxes.append(to_pil_image(img))

        return images_with_boxes

    def _filter_detection_output(self, predictions: dict, threshold: float) -> dict:
        """
        Filter detected results - labels, boxes, confidence scores based on threshold of score - leave if <= threshold

        :param predictions: output of model
        :param threshold: threshold of confidence to filter results
        :return: filtered predicted labels, boxes and confidence scores
        """
        # Process and filter output of model
        filtered_output = {
            'labels': [],
            'boxes': [],
            'scores': []
        }
        for output in predictions:
            labels, boxes, scores = [], [], []
            for ind, score in enumerate(output['scores']):
                if score >= threshold:
                    scores.append(score.item())
                    labels.append(self.model_classes[output["labels"][ind]])
                    boxes.append(output['boxes'][ind].tolist())

            filtered_output['scores'].append(scores)
            filtered_output['labels'].append(labels)
            filtered_output['boxes'].append(boxes)

        return filtered_output

    def run(self, image_paths: List[str] | str, score_threshold: float = 0.9, to_visualize: bool = True) -> (
            dict, list):
        """
        Run predictions of the model.

        :param to_visualize: to visualize results or not
        :param image_paths: path's to images or image on which detect objects
        :param score_threshold: threshold of confidence to filter results (leave if <= threshold)
        :return: predicted labels, boxes and confidence scores
        """
        # Read and preprocess images
        raw_images, batch = self._image_preprocess(image_paths)
        image_names = [path.split("\\")[-1] for path in image_paths]

        # Get output of model
        outputs = self.model(batch)

        # Process and filter output of model
        filtered_output = self._filter_detection_output(outputs, score_threshold)
        filtered_output['images'] = image_names

        # Draw boxes on input images
        images_with_boxes = self.draw_boxes(raw_images, filtered_output['boxes'], filtered_output['labels'])

        # Visualize detection results
        if to_visualize:
            for ind, im in enumerate(images_with_boxes):
                im.show(title=f'Image: {image_names[ind]}')

        return filtered_output, images_with_boxes

    @staticmethod
    def save_results(save_folder_path: str, predictions: dict, images: list):
        """
        Save results of prediction in 'save_folder_path' in such structure:
        - save_folder_path
            - drawn_boxes - folder with images
            - boxes_labels_scores.json - dict of predictions

        :param save_folder_path: path where to save results: json file and images with detected boxes
        :param predictions: dict of predictions(classes, confidence scores, boxes, image names)
         that would be saved in json file (boxes_labels_scores.json)
        :param images: images with drawn boxes to save in output folder drawn_boxes
        """
        # Save predictions in json file
        os.makedirs(save_folder_path, exist_ok=True)
        with open(os.path.join(save_folder_path, 'boxes_labels_scores.json'), "w") as out_json:
            json.dump(predictions, out_json)

        # Save images with drawn boxes
        dir_for_images = os.path.join(save_folder_path, 'drawn_boxes')
        os.makedirs(dir_for_images, exist_ok=True)

        for ind, img in enumerate(images):
            img.save(os.path.join(dir_for_images, predictions['images'][ind]))
