from yolov3_tf2.models import YoloV3Tiny, YoloV3
from yolov3_tf2.dataset import transform_images

from absl import flags

def handle_flags(**kwargs):
    flags.FLAGS(['program_name'])

    if kwargs is not None:
        for key, value in kwargs.items():
            setattr(flags.FLAGS, key, value)

def create_yolo_tiny(**kwargs):
    handle_flags(**kwargs)

    yolo = YoloV3Tiny(classes=80)
    yolo.load_weights('data/yolov3-tiny.tf').expect_partial()

    return yolo

def create_yolo(**kwargs):
    handle_flags(**kwargs)

    yolo = YoloV3(classes=80)
    yolo.load_weights('data/yolov3.tf').expect_partial()

    return yolo

