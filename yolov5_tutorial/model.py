# __example_code_start__
import torch
from PIL import Image
import numpy as np
from io import BytesIO
from fastapi.responses import Response
from fastapi import FastAPI

from ray import serve


@serve.deployment(
    ray_actor_options={"num_gpus": 1},
    autoscaling_config={"min_replicas": 1, "max_replicas": 2},
)
class ObjectDetection:
    def __init__(self):
        self.model = torch.hub.load("ultralytics/yolov5", "yolov5s")
        self.model.cuda()

    def detect(self, image_url: str):
        result_im = self.model(image_url)
        return Image.fromarray(result_im.render()[0].astype(np.uint8))


entrypoint = ObjectDetection.bind()
