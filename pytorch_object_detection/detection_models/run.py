import argparse
import os
import json
from pytorch_object_detection.detection_models.models import FasterRCNNDetection, RetinaNetDetection


def process_console_run(console_input: argparse.Namespace) -> None:
    """
    Logic of the console detection startup processing.
    :param console_input: output of console or dict with needed key values
    """
    # Select model
    match console_input.model_name:
        case 'Faster R-CNN':
            model = FasterRCNNDetection()
        case 'RetinaNet':
            model = RetinaNetDetection()
        case _:
            print('This model is not implemented')
            return

    # Get detection predictions
    image_names = os.listdir(path=console_input.source_folder)
    paths_to_images = [console_input.source_folder + '/' + name for name in image_names]
    predictions, images = model.run(
        paths_to_images,
        score_threshold=console_input.score_threshold
    )

    # Save predictions in json file
    if not os.path.exists(console_input.save_folder_path):
        os.mkdir(console_input.save_folder_path)
    out_json = open(console_input.save_folder_path + '/boxes_labels_scores.json', "w")
    json.dump(predictions, out_json)

    # Save images with drawn boxes
    dir_for_images = console_input.save_folder_path + '/drawn_boxes/'
    if not os.path.exists(dir_for_images):
        os.mkdir(dir_for_images)

    for ind, img in enumerate(images):
        img.save(dir_for_images + image_names[ind])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-folder', type=str, default='./test_data', help='path to folder with images, e.g. "./test_data" ')
    parser.add_argument('--model-name', type=str, default='Faster R-CNN', help='model name to use, available: '
                                                                               'Faster R-CNN, RetinaNet')
    parser.add_argument('--score-threshold', type=float, default=0.9, help='min confidence threshold of prediction,'
                                                                           ' e.g 0.8')
    parser.add_argument('--save-folder-path', type=str, default='./test_results', help='path to folder where to save predicted labels,'
                                                                   ' boxes, scores, images, e.g. "./test_results"')
    opt = parser.parse_args()
    process_console_run(opt)

