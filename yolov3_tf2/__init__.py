from yolov3_tf2.models import YoloV3Tiny
from yolov3_tf2.dataset import transform_images

from absl import flags


def create_yolo_tiny():
    flags.FLAGS(['program_name'])

    yolo = YoloV3Tiny(classes=80)
    yolo.load_weights('data/yolov3-tiny.tf').expect_partial()

    return yolo

