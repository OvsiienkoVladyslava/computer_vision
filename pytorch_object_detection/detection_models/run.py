from pytorch_object_detection.detection_models.models import FasterRCNNDetection

if __name__ == '__main__':
    path = "../test_data/cat_duck_dog.jpg"

    model = FasterRCNNDetection()
    labels, boxes, scores = model.run(path, score_threshold=0.5)
