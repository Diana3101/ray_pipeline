import requests
import rasterio
from shapely.geometry import Point
import json
import geopandas as gpd
import time
import os
import ray

from object_detector import ObjectDetector

with rasterio.open('batch/20220706_082111_SN16_L3_SR_MS-5844-13636-3-1.tiff') as image:
    image_array = image.read()
    image_crs = image.crs
    image_transform = image.transform

data = {'image_array': image_array.tolist(),
        'is_batching': False}

time_rest_0 = time.time()
response = requests.post("http://127.0.0.1:8000/", json=data)
time_rest_1 = time.time()
print(f'Time for REST: {time_rest_1-time_rest_0}')
results = response.json()

# print(results)

detector = ObjectDetector()
time_0 = time.time()
res = detector.detect(image_array)
time_1 = time.time()
print(f'Time simple: {time_1-time_0}')


batch_list = []
directory = 'batch'
for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    with rasterio.open(f) as image:
        image_array = image.read()
        batch_list.append(image_array)
        # image_crs = image.crs
        # image_transform = image.transform


@ray.remote
def send_query(image_array):
    data = {'image_array': image_array.tolist(),
            'is_batching': True}
    response = requests.post("http://127.0.0.1:8000/", json=data)
    return response.json()


time_rest_0 = time.time()
results = ray.get([send_query.remote(image_array) for image_array in batch_list])
time_rest_1 = time.time()
print(f'Time for REST (batch): {time_rest_1 - time_rest_0}')


time_0 = time.time()
res = detector.detect(batch_list)
time_1 = time.time()
print(f'Time simple (batch): {time_1-time_0}')







def results_to_geojson(results, crs=image_crs, transform=image_transform, output_format: str = 'jsonl'):
    geometries = []
    predictions = []
    predictions_proba = []
    ##########
    names = ["crater"]

    for i, result in enumerate(results):
        # print(result)

        for det in result:
            #### point
            x, y = [det['y']], [det['x']]
            x, y = rasterio.transform.xy(transform, x, y)
            point_geom = Point((x[0], y[0]))
            geometries.append(point_geom)

            predictions.append(names[det['cls']])
            predictions_proba.append(json.dumps({names[det['cls']]: float(det['conf'])}))

    geometries = gpd.GeoDataFrame(crs=crs, geometry=geometries)
    geometries['object'] = predictions
    geometries['object' + '_proba'] = predictions_proba

    geometries.to_crs(4326, inplace=True)

    if output_format == 'jsonl':
        geojson_output = '20221007_123156_SN20_L3_SR_MS-3896-19480-crater.geojsonl'
        data = json.loads(geometries.to_json())['features']
        output_data = [json.dumps(d) for d in data]

        with open(geojson_output, 'w') as fout:
            fout.write("\n".join(output_data))
    else:
        output_data = geometries
        # geojson_output = os.path.join(
        #     args.output_dir,
        #     dataset_name + '-crater.geojson')
        # geometries.to_file(geojson_output, driver="GeoJSON")

    # print(type(output_data))
    return output_data


# geojson_results = results_to_geojson(results)
# print(geojson_results)
