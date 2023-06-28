import time
import os
import requests

import rasterio
import ray

from object_detector import ObjectDetector


with rasterio.open('batch/20220706_082111_SN16_L3_SR_MS-5844-13636-3-1.tiff') as image:
    image_array = image.read()

data = {'image_array': image_array.tolist(),
        'is_batching': False}

time_rest_0 = time.time_ns() // 1_000_000
response = requests.post("http://127.0.0.1:8000/", json=data)
time_rest_1 = time.time_ns() // 1_000_000
print(f'Time for REST: {time_rest_1-time_rest_0} ms')
results = response.json()


detector = ObjectDetector()
time_0 = time.time_ns() // 1_000_000
res = detector.detect(image_array)
time_1 = time.time_ns() // 1_000_000
print(f'Time simple: {time_1-time_0} ms')


batch_list = []
directory = 'batch'
for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    with rasterio.open(f) as image:
        image_array = image.read()
        batch_list.append(image_array)


@ray.remote
def send_query(image_array):
    data = {'image_array': image_array.tolist(),
            'is_batching': True}
    response = requests.post("http://127.0.0.1:8000/", json=data)
    return response.json()


time_rest_0 = time.time_ns() // 1_000_000
results = ray.get([send_query.remote(image_array) for image_array in batch_list])
time_rest_1 = time.time_ns() // 1_000_000
print(f'Time for REST (batch): {time_rest_1 - time_rest_0} ms')


time_0 = time.time_ns() // 1_000_000
res = detector.detect(batch_list)
time_1 = time.time_ns() // 1_000_000
print(f'Time simple (batch): {time_1-time_0} ms')
