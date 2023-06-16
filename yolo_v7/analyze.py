import argparse
import logging
import os
import shutil
from glob import glob
import json

import cv2
import numpy as np
import rasterio
from rasterio.windows import Window
from shapely.geometry import Polygon, Point
import geopandas as gpd

import torch
from models.experimental import attempt_load
from utils.torch_utils import select_device
from utils.general import (
    check_img_size, 
    non_max_suppression,
    scale_coords,
    xyxy2xywh
)

from tqdm.notebook import tqdm, trange



logger = logging.getLogger("Building Footprint Detection")
logger.setLevel(logging.DEBUG)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', required=True, type=str, help='Path to input image')
    parser.add_argument('--output_dir', required=True, type=str, help='Path to directory with outputs')
    parser.add_argument('--model_path', type=str, required=True, help='Path to model')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--img_size', type=int, default=512)
    parser.add_argument('--output_format', required=False, type=str, default='jsonl', help='Format of output file, default jsonl')
    parser.add_argument('--output_label', required=False, type=str, default='object', help='Output field name with results')
    parser.add_argument('--device', required=False, type=str, default='cpu', help='Device')

    args, unknown = parser.parse_known_args()
    return args


def detect(batch, model, conf_thres=0.3, iou_thres=0.5, device='cpu', half=False, classes=[0], agnostic_nms = False):
    batch = np.array(batch)
    batch = batch.transpose(0, 3, 1, 2)  # BGR to RGB, to 3x416x416

    batch = torch.from_numpy(batch).to(device)
    batch = batch.half() if half else batch.float()  # uint8 to fp16/32
    batch /= 255.0  # 0 - 255 to 0.0 - 1.0
    if batch.ndimension() == 3:
        batch = batch.unsqueeze(0)

    with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
        pred = model(batch, augment=False)[0]

    # Apply NMS
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms)

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



def process(args):
    input_file = args.input_file
    output_dir = args.output_dir
    img_size = args.img_size
    batch_size = args.batch_size

    dataset_name = input_file.split("/")[-1].split(".")[0]


    names = ["crater"]
    device = select_device(args.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA
    # Load model
    model = attempt_load(args.model_path, map_location=device)  # load FP32 model


    with rasterio.open(input_file) as data:
        profile = data.profile
        transform = data.transform

        def batch_generator():
            image_batch = []
            transform_batch = []
            for i in trange(profile['width'] // img_size):
                for j in range(profile['height'] // img_size):
                    win = Window(img_size * i, img_size * j, img_size, img_size)
                    
                    if len(image_batch) == batch_size:
                        image_batch = []
                        transform_batch = []

                    img = data.read( window=win, out_dtype=np.uint8).transpose(1, 2, 0)
                    # print('image')

                    if img.ndim == 4:
                        img[img[..., 3] == 0] = 0
                    if img.shape[-1] == 4:
                        img = img[..., :3]

                    trn = data.window_transform(win)
                    image_batch.append(img)
                    transform_batch.append(trn)

                    if len(image_batch) == batch_size:
                        yield (image_batch, transform_batch)
            yield (image_batch, transform_batch)

        gen = batch_generator()
        geometries = []
        predictions = []
        types = []
        predictions_proba = []
        for batch_id, batch in tqdm(enumerate(gen)):

            # results = model.predict(batch[0], conf=0.1, iou=0.5, max_det=600)
            results = detect(batch[0], model, device=device)

            for i, result in enumerate(results):
                print(result)
                
                img = batch[0][i]
                transform = batch[1][i]
                det_ind = 0
                for det in result:
                    #### bbox
                    # xs = np.array([det['xmin'], det['xmin'], det['xmax'], det['xmax']])
                    # ys = np.array([det['ymin'], det['ymax'], det['ymax'], det['ymin']])

                    # xs, ys = rasterio.transform.xy(transform, xs, ys)
                    # lons= np.array(xs)
                    # lats = np.array(ys)

                    # polygon_geom = Polygon(zip(lons, lats))
                    # geometries.append(polygon_geom)
                    #### point

                    x, y = [det['y']], [det['x']]
                    x, y = rasterio.transform.xy(transform, x, y)
                    point_geom = Point((x[0], y[0]))
                    geometries.append(point_geom)

                    predictions.append(names[det['cls']])
                    predictions_proba.append(json.dumps({names[det['cls']]: float(det['conf'])}))

        geometries = gpd.GeoDataFrame(crs=data.crs, geometry=geometries)
        geometries[args.output_label] = predictions
        geometries[args.output_label + '_proba'] = predictions_proba

        geometries.to_crs(4326, inplace=True)

        if args.output_format == 'jsonl':
            geojson_output = os.path.join(
                args.output_dir, 
                dataset_name + '-crater.geojsonl')
            data = json.loads(geometries.to_json())['features']
            data = [json.dumps(d) for d in data]
            with open(geojson_output, 'w') as fout:
                fout.write("\n".join(data))
        else:
            geojson_output = os.path.join(
                args.output_dir, 
                dataset_name + '-crater.geojson')
            geometries.to_file(geojson_output, driver="GeoJSON")  




def main(args):
    input_file = args.input_file
    dataset_name = input_file.split("/")[-1].split(".")[0]

    logger.info(f"Processing started")
    process(args)    



if __name__ == "__main__":
    args = parse_args()
    main(args)