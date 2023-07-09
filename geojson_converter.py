import os
import json

import rasterio
from shapely.geometry import Polygon, Point
import geopandas as gpd
import pandas as pd
from tqdm import tqdm


class GeoJsonConverter:
    def __init__(self, output_format, output_dir, name, output_label, united_geojson):
        self.output_format = output_format
        self.output_dir = output_dir
        self.names = [name]
        self.output_label = output_label
        self.united_geojson = united_geojson

    def transform_result(self, result, transform,
                         geometries, predictions, predictions_proba):

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

            predictions.append(self.names[det['cls']])
            predictions_proba.append(json.dumps({self.names[det['cls']]: float(det['conf'])}))

        return geometries, predictions, predictions_proba

    def save_file(self, output_file_name, geometries):
        if self.output_format == 'jsonl':
            geojson_output = os.path.join(
                self.output_dir,
                output_file_name + 'crater.geojsonl')
            data = json.loads(geometries.to_json())['features']
            data = [json.dumps(d) for d in data]
            with open(geojson_output, 'w') as fout:
                fout.write("\n".join(data))
        else:
            geojson_output = os.path.join(
                self.output_dir,
                output_file_name + 'crater.geojson')
            geometries.to_file(geojson_output, driver="GeoJSON")

    def results_to_geotiff(self, results, batch, lens_image_batch):
        j = 0
        geometries = []
        predictions = []
        predictions_proba = []

        if self.united_geojson:
            united_geometries = gpd.GeoDataFrame(pd.DataFrame())

        for i, result_per_image in tqdm(enumerate(results)):
            max_i = sum(lens_image_batch[:j + 1])
            idx = i if j == 0 else i - sum(lens_image_batch[:j])

            if i < max_i:
                geometries, predictions, predictions_proba = self.transform_result(result=result_per_image,
                                                                                   transform=batch[1][j][idx],
                                                                                   geometries=geometries,
                                                                                   predictions=predictions,
                                                                                   predictions_proba=predictions_proba)

            if i == max_i or i == (len(results) - 1):
                geometries = gpd.GeoDataFrame(crs=batch[2][j], geometry=geometries)
                geometries[self.output_label] = predictions
                geometries[self.output_label + '_proba'] = predictions_proba

                geometries.to_crs(4326, inplace=True)

                if self.united_geojson:
                    united_geometries = pd.concat([united_geometries, geometries],
                                                  ignore_index=True)
                else:
                    output_file_name = batch[3][j] + '-'
                    self.save_file(output_file_name=output_file_name,
                                   geometries=geometries)

                if i != (len(results) - 1):
                    j += 1
                    idx = i - sum(lens_image_batch[:j])

                    geometries = []
                    predictions = []
                    predictions_proba = []

                    geometries, predictions, predictions_proba = self.transform_result(result_per_image,
                                                                                       batch[1][j][idx],
                                                                                       geometries,
                                                                                       predictions,
                                                                                       predictions_proba)
        if self.united_geojson:
            self.save_file(output_file_name='',
                           geometries=united_geometries)
