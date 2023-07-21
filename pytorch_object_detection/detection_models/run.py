import argparse
import os
import json

from pytorch_object_detection.detection_models.configs import DetectionModelsNames


def process_console_run(
        source_folder: str,
        model_name: str,
        score_threshold: float,
        save_folder_path: str,
        **kwargs
):
    """
    Logic of the console detection startup processing.

    :param source_folder:
    :param model_name:
    :param score_threshold:
    :param save_folder_path:
    """
    # Select and initialize model
    model = DetectionModelsNames.from_string(model_name)()

    # Get detection predictions
    image_names = os.listdir(path=source_folder)
    paths_to_images = [os.path.join(source_folder, name) for name in image_names]
    predictions, images = model.run(
        paths_to_images,
        score_threshold=score_threshold
    )

    # Save predictions in json file
    os.makedirs(save_folder_path, exist_ok=True)
    with open(os.path.join(save_folder_path, 'boxes_labels_scores.json'), "w") as out_json:
        json.dump(predictions, out_json)

    # Save images with drawn boxes
    dir_for_images = os.path.join(save_folder_path, 'drawn_boxes')
    os.makedirs(dir_for_images, exist_ok=True)

    for ind, img in enumerate(images):
        img.save(os.path.join(dir_for_images, image_names[ind]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Run inference of selected detection model')
    parser.add_argument('--source-folder', type=str, default='./test_data',
                        help='path to folder with images, e.g. "./test_data" ')
    parser.add_argument('--model-name', type=str, default='Faster RCNN',
                        help='model name to use, available: Faster RCNN, RetinaNet')
    parser.add_argument('--score-threshold', type=float, default=0.9,
                        help='min confidence threshold of prediction, e.g 0.8')
    parser.add_argument('--save-folder-path', type=str, default='./test_results',
                        help='path to folder where to save predicted labels, boxes, scores, images,'
                             ' e.g. "./test_results"')
    opt = parser.parse_args()
    process_console_run(**vars(opt))

