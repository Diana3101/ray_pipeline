import torch
from ray import serve
import numpy as np
from starlette.requests import Request

from yolo_v7.models.experimental import attempt_load
from yolo_v7.utils.torch_utils import select_device
from yolo_v7.utils.general import (non_max_suppression, xyxy2xywh)


@serve.deployment(
    ray_actor_options={"num_gpus": 1, "num_cpus": 6},
    autoscaling_config={"min_replicas": 1, "max_replicas": 4}
)
class ObjectDetector:
    def __init__(self, weights='/data/dianakapatsyn/ray_pipeline/yolo_v7/detection-v3.pt'):
        # Load model
        print('-------Load model----------')
        self.model = attempt_load(weights, map_location=torch.device('cpu'))  # load FP32 model
        print('-------Finish----------')
        print('Is CUDA available: ', torch.cuda.is_available())

        # Initialize
        self.model.cuda()
        self.device = torch.device('cuda:0')
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA
        self.model.eval()
        print(f'DEVICE: {self.device}')

        if self.half:
            self.model.half()  # to FP16

    def detect(self, batch):
        if isinstance(batch, list):
            batch = np.array(batch)

        batch = torch.from_numpy(batch).to(self.device)
        batch = batch.half() if self.half else batch.float()  # uint8 to fp16/32
        batch /= 255.0  # 0 - 255 to 0.0 - 1.0

        if batch.ndimension() == 3:
            batch = batch.unsqueeze(0)

        with torch.no_grad():  # Calculating gradients would cause a GPU memory leak
            pred = self.model(batch, augment=False)[0]

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres=0.3, iou_thres=0.5, classes=[0], agnostic=False)

        # Process detections
        results = []
        for i, det in enumerate(pred):  # detections per image
            res = []

            if len(det):
                # # Write results

                for *xyxy, conf, cls in reversed(det):
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4))).view(-1).tolist()  # normalized xywh

                    res.append({
                        'x': xywh[0],
                        'y': xywh[1],
                        'w': xywh[2],
                        'h': xywh[3],
                        'xmin': xyxy[0].item(),
                        'ymin': xyxy[1].item(),
                        'xmax': xyxy[2].item(),
                        'ymax': xyxy[3].item(),
                        'conf': float(conf),
                        'cls': int(cls)
                    })
            results.append(res)

        return results

    @serve.batch(max_batch_size=32,
                 # time to wait before returning an incomplete batch (in seconds)
                 batch_wait_timeout_s=0.5)
    async def handle_batch(self, input_batch: np.ndarray):
        print("Our input batch has length:", len(input_batch))

        results = self.detect(batch=input_batch)
        return results

    async def __call__(self, http_request: Request):
        data = await http_request.json()
        image_array = np.array(data['image_array'])
        is_batching = data['is_batching']

        if is_batching:
            return await self.handle_batch(input_batch=image_array)

        return self.detect(batch=image_array)


detector = ObjectDetector.bind()
