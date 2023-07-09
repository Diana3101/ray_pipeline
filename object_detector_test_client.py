import time
import os
import requests

import rasterio

batch_list = []
directory = 'batch'
for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    with rasterio.open(f) as image:
        image_array = image.read()
        batch_list.append(image_array.tolist())

# batch_list_big = batch_list
# batch_list_big.extend(batch_list)


@ray.remote
def send_query(image_array):
    data = {'image_array': image_array,
            'is_batching': True}
    response = requests.post("http://127.0.0.1:8000/", json=data)
    return response.json()


time_rest_0 = time.time_ns() // 1_000_000
response = requests.post("http://127.0.0.1:8000/", json=data)
time_rest_1 = time.time_ns() // 1_000_000
print(f'Time for {len(batch_list)}-batch using batch_handler: {time_rest_1 - time_rest_0} ms')

data = {'image_array': batch_list[0],
        'is_batching': False}

time_rest_0 = time.time_ns() // 1_000_000
response = requests.post("http://127.0.0.1:8000/", json=data)
time_rest_1 = time.time_ns() // 1_000_000
print(f'Time for one image without batch_handler: {time_rest_1 - time_rest_0} ms')
results = response.json()
print(results)
