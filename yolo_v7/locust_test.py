import os

from locust import HttpUser, task, constant_throughput
import rasterio
import ray

# with rasterio.open('batch/20220706_082111_SN16_L3_SR_MS-5844-13636-3-1.tiff') as image:
#     image_array = image.read()
#
# data = {'image_array': image_array.tolist(),
#         'is_batching': False}

batch_list = []
directory = 'batch'
for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    with rasterio.open(f) as image:
        image_array = image.read()
        batch_list.append(image_array.tolist())

data = {'image_array': batch_list,
        'is_batching': False}


class CraterLocation(HttpUser):
    # Means that a user will send 1 request per second
    wait_time = constant_throughput(1)

    # Task to be performed (send data & get response)
    @task
    def predict(self):
        self.client.post(
            "/",
            json=data
            # timeout=7,
        )
