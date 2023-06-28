from models.experimental import attempt_load
from utils.torch_utils import select_device
import torch
from utils.general import (non_max_suppression, xyxy2xywh)
import numpy as np


class ObjectDetector:
    def __init__(self, device='cpu', weights='detection-v3.pt'):
        # Initialize
        self.device = select_device(device)
        print(f'DEVICE: {self.device}')
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        print('-------Load model----------')
        self.model = attempt_load(weights, map_location=device)  # load FP32 model
        print('-------Finish----------')

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
        print('Batch shape before pred: ', batch.shape)

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
                        'xmin': xyxy[0],
                        'ymin': xyxy[1],
                        'xmax': xyxy[2],
                        'ymax': xyxy[3],
                        'conf': float(conf),
                        'cls': int(cls)
                    })
            results.append(res)

        return results
