import torchvision
import torch
from torchvision.io.image import read_image
from torchvision.models import WeightsEnum

from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image


class DetectionPipeline:
    model: torchvision.models.detection
    weights: WeightsEnum
    preprocess_pipeline: torchvision.transforms._presets.ObjectDetection
    model_classes: list

    def __init__(self):
        self.model.eval()
        self.model_classes = self.weights.meta["categories"]
        self.preprocess_pipeline = self.weights.transforms()

    def get_model_info(self):
        pass

    def _image_preprocess(self, image_paths: list[str] | str):
        if type(image_paths) is str:
            image_paths = [image_paths]

        raw_images = [read_image(path) for path in image_paths]
        batch = [self.preprocess_pipeline(img) for img in raw_images]

        return raw_images, batch

    @staticmethod
    def visualize_result(imgs, boxes, labels):
        for ind, img in enumerate(imgs):
            box = torch.stack(boxes[ind], dim=0)
            drawed_boxes = draw_bounding_boxes(img, boxes=box,
                                               labels=labels[ind],
                                               width=2)
            im = to_pil_image(drawed_boxes.detach())
            im.show(title=f'Image №{ind}')

    def run(self, imagepaths: list[str] | str, score_threshold=0.9):
        # Read and preprocess images
        raw_images, batch = self._image_preprocess(imagepaths)

        # Get output of model
        outputs = self.model(batch)

        # Process and filter output of model
        output_labels, output_boxes, output_scores = [], [], []
        for output in outputs:
            labels, boxes, scores = [], [], []
            for ind, score in enumerate(output['scores']):
                if score >= score_threshold:
                    scores.append(score)
                    labels.append(self.model_classes[output["labels"][ind]])
                    boxes.append(output['boxes'][ind])
            output_scores.append(scores)
            output_labels.append(labels)
            output_boxes.append(boxes)

        # Visualize detection results
        self.visualize_result(raw_images, output_boxes, output_labels)

        return output_labels, output_boxes, output_scores