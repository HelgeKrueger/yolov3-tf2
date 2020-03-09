from yolov3_tf2.models import YoloV3Tiny, YoloV3
from yolov3_tf2.dataset import transform_images

from absl import flags


def handle_flags(**kwargs):
    flags.FLAGS(['program_name'])

    if kwargs is not None:
        for key, value in kwargs.items():
            setattr(flags.FLAGS, key, value)


def create_yolo_tiny(weights='checkpoints/yolov3-tiny.tf', classes=80, **kwargs):
    handle_flags(**kwargs)

    yolo = YoloV3Tiny(classes=classes)
    yolo.load_weights(weights).expect_partial()

    return yolo


def create_yolo(weights='checkpoints/yolov3.tf', classes=80, **kwargs):
    handle_flags(**kwargs)

    yolo = YoloV3(classes=classes)
    yolo.load_weights(weights).expect_partial()

    return yolo
