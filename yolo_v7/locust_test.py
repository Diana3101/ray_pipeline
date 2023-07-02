import os

from locust import HttpUser, task, constant_throughput
import rasterio
import ray

batch_list = []
directory = 'batch'
for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    with rasterio.open(f) as image:
        image_array = image.read()
        batch_list.append(image_array.tolist())


class CraterLocation(HttpUser):
    # Means that a user will send 1 request per second
    wait_time = constant_throughput(1)

    # Task to be performed (send data & get response)
    @task
    def predict(self):
        for image_array in batch_list:
            data = {'image_array': image_array,
                    'is_batching': True}
            self.client.post(
                "/",
                json=data,
                timeout=5
            )

